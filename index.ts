import {
	BedrockRuntimeClient,
	InvokeModelCommand,
	InvokeModelWithResponseStreamCommand,
} from "@aws-sdk/client-bedrock-runtime";

const bedrockClient = new BedrockRuntimeClient({
	region: process.env.AWS_REGION || "us-west-2",
});

const TINYBIRD_BASE_URL = process.env.TINYBIRD_BASE_URL;
const TINYBIRD_TOKEN = process.env.TINYBIRD_TOKEN;
const tinybirdEnabled = !!(TINYBIRD_BASE_URL && TINYBIRD_TOKEN);

// Provider configuration
interface ProviderConfig {
	name: string;
	baseUrl: string;
	apiKeyEnvVar: string;
	responsesEndpoint: string;
}

const PROVIDERS = {
	openai: {
		name: "OpenAI",
		baseUrl: "https://api.openai.com",
		apiKeyEnvVar: "OPENAI_API_KEY",
		responsesEndpoint: "/v1/responses",
	},
	// Add more providers here as needed
	// anthropic: {
	//   name: "Anthropic",
	//   baseUrl: "https://api.anthropic.com",
	//   apiKeyEnvVar: "ANTHROPIC_API_KEY",
	//   responsesEndpoint: "/v1/messages",
	// },
} as const satisfies Record<string, ProviderConfig>;

// Determine provider based on model ID
function getProviderForModel(modelId: string): ProviderConfig {
	// Route based on model prefix
	if (
		modelId.startsWith("gpt-") ||
		modelId.startsWith("o1") ||
		modelId.startsWith("o3")
	) {
		return PROVIDERS.openai;
	}
	// Add more routing rules as needed
	// if (modelId.startsWith("claude-")) {
	//   return PROVIDERS.anthropic;
	// }

	// Default to OpenAI
	return PROVIDERS.openai;
}

// Call the provider's API
async function callProviderAPI(
	provider: ProviderConfig,
	body: Record<string, unknown>,
): Promise<Response> {
	const apiKey = process.env[provider.apiKeyEnvVar];
	if (!apiKey) {
		return Response.json(
			{
				error: {
					message: `${provider.apiKeyEnvVar} not configured`,
					type: "server_error",
				},
			},
			{ status: 500 },
		);
	}

	const url = `${provider.baseUrl}${provider.responsesEndpoint}`;

	return fetch(url, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			Authorization: `Bearer ${apiKey}`,
		},
		body: JSON.stringify(body),
	});
}

async function saveLog(
	entry: {
		modelId: string;
		request: any;
		response: any;
		streamed: boolean;
		durationMs: number;
		requestId?: string;
	},
	type: "openai" | "anthropic" = "anthropic",
) {
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

	const datasourceName =
		type === "openai"
			? "openai_inference_events"
			: "anthropic_inference_events";

	try {
		const res = await fetch(
			`${TINYBIRD_BASE_URL}/v0/events?name=${datasourceName}`,
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

function log(direction: "REQ" | "RES", modelId: string, body: string | object) {
	const timestamp = new Date().toISOString();
	const parsed = typeof body === "string" ? JSON.parse(body) : body;
	console.log(`\n[${timestamp}] ${direction} ${modelId}`);
	console.log(JSON.stringify(parsed, null, 2));
}

// Bedrock invoke handler
async function handleBedrockInvoke(req: Request) {
	const url = new URL(req.url);
	const match = url.pathname.match(/^\/model\/(.+)\/invoke$/);
	if (!match) return new Response("Not found", { status: 404 });

	const modelId = decodeURIComponent(match[1] ?? "");
	const body = await req.text();
	const startTime = Date.now();

	log("REQ", modelId, body);

	const command = new InvokeModelCommand({
		modelId,
		contentType: "application/json",
		accept: "application/json",
		body,
	});

	const response = await bedrockClient.send(command);
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

// Bedrock streaming handler
async function handleBedrockStream(req: Request) {
	const url = new URL(req.url);
	const match = url.pathname.match(
		/^\/model\/(.+)\/invoke-with-response-stream$/,
	);
	if (!match) return new Response("Not found", { status: 404 });

	const modelId = decodeURIComponent(match[1] ?? "");
	const body = await req.text();
	const startTime = Date.now();

	log("REQ", modelId, body);

	const command = new InvokeModelWithResponseStreamCommand({
		modelId,
		contentType: "application/json",
		accept: "application/json",
		body,
	});

	const response = await bedrockClient.send(command);
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
					cache_read_input_tokens: messageStart?.usage?.cache_read_input_tokens,
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

// Responses API handler - routes to provider based on model ID
async function handleResponses(req: Request) {
	const body = (await req.json()) as Record<string, unknown>;
	const startTime = Date.now();
	const modelId = body.model as string;
	const isStreaming = body.stream === true;

	// Get provider based on model ID
	const provider = getProviderForModel(modelId);

	log("REQ", `${provider.name}/${modelId}`, body);

	// Call the provider's API
	const providerResponse = await callProviderAPI(provider, body);

	// Check if callProviderAPI returned an error response (missing API key)
	if (
		providerResponse.headers.get("content-type")?.includes("application/json")
	) {
		const cloned = providerResponse.clone();
		try {
			const json = (await cloned.json()) as { error?: unknown };
			if (json.error) {
				return providerResponse;
			}
		} catch {
			// Not JSON or not an error, continue
		}
	}

	if (!providerResponse.ok) {
		const errorBody = await providerResponse.text();
		console.error(
			`[${provider.name} Error] ${providerResponse.status}: ${errorBody}`,
		);
		return new Response(errorBody, {
			status: providerResponse.status,
			headers: { "content-type": "application/json" },
		});
	}

	if (isStreaming) {
		// For streaming, we need to tee the stream to log while proxying
		const [logStream, proxyStream] = providerResponse.body!.tee();

		// Log stream chunks in background
		(async () => {
			const reader = logStream.getReader();
			const decoder = new TextDecoder();
			let fullContent = "";
			let lastEvent: any = null;

			try {
				while (true) {
					const { done, value } = await reader.read();
					if (done) break;

					const text = decoder.decode(value, { stream: true });
					const lines = text
						.split("\n")
						.filter((line) => line.startsWith("data: "));

					for (const line of lines) {
						const data = line.slice(6); // Remove "data: " prefix
						if (data === "[DONE]") continue;

						try {
							const event = JSON.parse(data);
							lastEvent = event;
							// Responses API uses output_text.delta for text content
							if (event.type === "response.output_text.delta") {
								fullContent += event.delta ?? "";
							}
						} catch {
							// Ignore parse errors for partial chunks
						}
					}
				}

				log("RES", `${provider.name}/${modelId}`, {
					content: fullContent,
					lastEvent,
				});
				await saveLog(
					{
						modelId,
						request: body,
						response: { content: fullContent, lastEvent },
						streamed: true,
						durationMs: Date.now() - startTime,
					},
					"openai",
				);
			} catch (err) {
				console.error("[Stream logging error]", err);
			}
		})();

		return new Response(proxyStream, {
			headers: {
				"content-type": "text/event-stream",
				"cache-control": "no-cache",
				connection: "keep-alive",
			},
		});
	}

	// Non-streaming: parse, log, and return
	const responseBody = (await providerResponse.json()) as Record<
		string,
		unknown
	>;

	log("RES", `${provider.name}/${modelId}`, responseBody);

	await saveLog(
		{
			modelId,
			request: body,
			response: responseBody,
			streamed: false,
			durationMs: Date.now() - startTime,
		},
		"openai",
	);

	return Response.json(responseBody);
}

const server = Bun.serve({
	port: 7070,
	routes: {
		// Responses API endpoint - routes to provider based on model ID
		"/v1/responses": {
			POST: handleResponses,
		},

		// Bedrock proxy endpoints (using wildcard for dynamic model IDs)
		"/model/*/invoke": {
			POST: handleBedrockInvoke,
		},
		"/model/*/invoke-with-response-stream": {
			POST: handleBedrockStream,
		},
	},

	// Fallback for unmatched routes
	fetch(req) {
		console.log(`[404] ${req.method} ${new URL(req.url).pathname}`);
		return new Response("Not found", { status: 404 });
	},
});

console.log(`Proxy server running on http://localhost:${server.port}`);
console.log(`  - Bedrock: /model/{modelId}/invoke`);
console.log(`  - Bedrock: /model/{modelId}/invoke-with-response-stream`);
console.log(`  - OpenAI:  /v1/responses`);
if (tinybirdEnabled) {
	console.log(`Tinybird logging enabled`);
}
