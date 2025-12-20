import {
	BedrockRuntimeClient,
	InvokeModelCommand,
	InvokeModelWithResponseStreamCommand,
} from "@aws-sdk/client-bedrock-runtime";

const client = new BedrockRuntimeClient({
	region: process.env.AWS_REGION || "us-west-2",
});

const TINYBIRD_BASE_URL = process.env.TINYBIRD_BASE_URL;
const TINYBIRD_TOKEN = process.env.TINYBIRD_TOKEN;
const tinybirdEnabled = !!(TINYBIRD_BASE_URL && TINYBIRD_TOKEN);

async function saveLog(entry: {
	modelId: string;
	request: any;
	response: any;
	streamed: boolean;
	durationMs: number;
	requestId?: string;
}) {
	if (!tinybirdEnabled) return;

	const row = {
		timestamp: new Date().toISOString(),
		conversation_query_run_id: process.env.CONVERSATION_QUERY_RUN_ID ?? "",
		data: JSON.stringify({
			modelId: entry.modelId,
			request: entry.request,
			response: entry.response,
			streamed: entry.streamed,
			durationMs: entry.durationMs,
			requestId: entry.requestId,
		}),
	};

	try {
		const res = await fetch(
			`${TINYBIRD_BASE_URL}/v0/events?name=anthropic_inference_events`,
			{
				method: "POST",
				headers: {
					Authorization: `Bearer ${TINYBIRD_TOKEN}`,
					"Content-Type": "application/json",
				},
				body: JSON.stringify(row),
			},
		);
		if (!res.ok) {
			console.error(`[Tinybird] ${res.status}: ${await res.text()}`);
		}
	} catch (err) {
		console.error(`[Tinybird] Error:`, err);
	}
}

function log(direction: "REQ" | "RES", modelId: string, body: string) {
	const timestamp = new Date().toISOString();
	const parsed = JSON.parse(body);
	console.log(`\n[${timestamp}] ${direction} ${modelId}`);
	console.log(JSON.stringify(parsed, null, 2));
}

const server = Bun.serve({
	port: 7070,
	async fetch(req) {
		const url = new URL(req.url);
		const pathname = url.pathname;

		// Match /model/{modelId}/invoke
		const invokeMatch = pathname.match(/^\/model\/(.+)\/invoke$/);
		if (invokeMatch && req.method === "POST") {
			const modelId = decodeURIComponent(invokeMatch[1] ?? "");
			const body = await req.text();
			const startTime = Date.now();

			log("REQ", modelId, body);

			const command = new InvokeModelCommand({
				modelId,
				contentType: "application/json",
				accept: "application/json",
				body,
			});

			const response = await client.send(command);
			const responseBody = new TextDecoder().decode(response.body);

			log("RES", modelId, responseBody);

			await saveLog({
				modelId,
				request: JSON.parse(body),
				response: JSON.parse(responseBody),
				streamed: false,
				durationMs: Date.now() - startTime,
				requestId: response.$metadata.requestId,
			});

			return new Response(responseBody, {
				headers: { "content-type": "application/json" },
			});
		}

		// Match /model/{modelId}/invoke-with-response-stream
		const streamMatch = pathname.match(
			/^\/model\/(.+)\/invoke-with-response-stream$/,
		);
		if (streamMatch && req.method === "POST") {
			const modelId = decodeURIComponent(streamMatch[1] ?? "");
			const body = await req.text();
			const startTime = Date.now();

			log("REQ", modelId, body);

			const command = new InvokeModelWithResponseStreamCommand({
				modelId,
				contentType: "application/json",
				accept: "application/json",
				body,
			});

			const response = await client.send(command);
			const requestId = response.$metadata.requestId;
			const chunks: any[] = [];

			const stream = new ReadableStream({
				async start(controller) {
					if (response.body) {
						for await (const event of response.body) {
							if (event.chunk?.bytes) {
								const chunkStr = new TextDecoder().decode(event.chunk.bytes);
								chunks.push(JSON.parse(chunkStr));
								controller.enqueue(event.chunk.bytes);
							}
						}
					}
					// Reconstruct full message from stream chunks
					const messageStart = chunks.find(
						(c) => c.type === "message_start",
					)?.message;
					const contentDeltas = chunks
						.filter((c) => c.type === "content_block_delta")
						.map((c) => c.delta?.text ?? "")
						.join("");
					const messageDelta = chunks.find((c) => c.type === "message_delta");

					const reconstructed = {
						model: messageStart?.model,
						id: messageStart?.id,
						type: "message",
						role: messageStart?.role,
						content: [{ type: "text", text: contentDeltas }],
						stop_reason: messageDelta?.delta?.stop_reason,
						stop_sequence: messageDelta?.delta?.stop_sequence ?? null,
						usage: {
							input_tokens: messageStart?.usage?.input_tokens,
							cache_creation_input_tokens:
								messageStart?.usage?.cache_creation_input_tokens,
							cache_read_input_tokens:
								messageStart?.usage?.cache_read_input_tokens,
							output_tokens: messageDelta?.usage?.output_tokens,
						},
					};

					console.log(`\n[${new Date().toISOString()}] RES ${modelId}`);
					console.log(JSON.stringify(reconstructed, null, 2));

					await saveLog({
						modelId,
						request: JSON.parse(body),
						response: reconstructed,
						streamed: true,
						durationMs: Date.now() - startTime,
						requestId,
					});

					controller.close();
				},
			});

			return new Response(stream, {
				headers: { "content-type": "application/vnd.amazon.eventstream" },
			});
		}

		console.log(`[404] ${req.method} ${pathname}`);
		return new Response("Not found", { status: 404 });
	},
});

console.log(`Bedrock proxy running on http://localhost:${server.port}`);
if (tinybirdEnabled) {
	console.log(`Tinybird logging enabled`);
}
