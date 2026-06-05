export const COOKIE = "dash_auth";
const MSG = "axiom-dash-v1";

async function hmac(password: string): Promise<string> {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw", enc.encode(password), { name: "HMAC", hash: "SHA-256" }, false, ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(MSG));
  return btoa(String.fromCharCode(...new Uint8Array(sig)));
}

export async function tokenFor(): Promise<string> {
  const pw = process.env.DASHBOARD_PASSWORD ?? "";
  return hmac(pw);
}
export async function verifyToken(token: string | null | undefined): Promise<boolean> {
  if (!token) return false;
  const expected = await tokenFor();
  // constant-time-ish compare
  if (token.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < token.length; i++) diff |= token.charCodeAt(i) ^ expected.charCodeAt(i);
  return diff === 0;
}
