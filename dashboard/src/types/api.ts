export interface ApiResponse<T> {
  readonly data: T | null;
  readonly error: string | null;
  readonly timestamp: number;
}
