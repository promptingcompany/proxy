import { describe, expect, mock, test } from "bun:test";
import { configureStreamingRequestTimeout } from "./streaming-timeout";

describe("configureStreamingRequestTimeout", () => {
	test("disables Bun's request timeout for streaming responses", () => {
		const request = new Request("http://localhost/v1/responses");
		const timeout = mock(() => undefined);

		configureStreamingRequestTimeout(request, { timeout }, true);

		expect(timeout).toHaveBeenCalledTimes(1);
		expect(timeout).toHaveBeenCalledWith(request, 0);
	});

	test("keeps Bun's default timeout for non-streaming responses", () => {
		const request = new Request("http://localhost/v1/responses");
		const timeout = mock(() => undefined);

		configureStreamingRequestTimeout(request, { timeout }, false);

		expect(timeout).not.toHaveBeenCalled();
	});
});
