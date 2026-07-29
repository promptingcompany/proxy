export type RequestTimeoutServer = Pick<Bun.Server<undefined>, "timeout">;

/**
 * Let the downstream streaming client own its idle deadline.
 *
 * Bun's default per-request idle timeout also applies while a response is
 * streaming, so a quiet model/tool interval can otherwise close a healthy
 * upstream stream before the client receives its next event.
 */
export function configureStreamingRequestTimeout(
	request: Request,
	server: RequestTimeoutServer,
	isStreaming: boolean,
): void {
	if (isStreaming) {
		server.timeout(request, 0);
	}
}
