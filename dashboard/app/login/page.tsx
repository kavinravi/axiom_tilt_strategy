"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

export default function Login() {
  const router = useRouter();
  const [pw, setPw] = useState("");
  const [err, setErr] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(false);
    try {
      const res = await fetch("/api/login", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ password: pw }),
      });
      if (res.ok) router.replace("/now");
      else setErr(true);
    } catch {
      setErr(true);
    }
  }

  return (
    <main className="flex min-h-screen items-center justify-center p-6">
      <form onSubmit={submit} className="w-full max-w-xs space-y-4">
        <h1 className="text-xl font-semibold">Axiom Tilt</h1>
        <input
          type="password" value={pw} onChange={(e) => setPw(e.target.value)}
          placeholder="Password" aria-label="Password"
          className="w-full rounded-md bg-neutral-900 px-3 py-2 outline-none ring-1 ring-neutral-700 focus:ring-neutral-400"
        />
        {err && <p className="text-sm text-red-400">Wrong password.</p>}
        <button type="submit" className="w-full rounded-md bg-neutral-100 px-3 py-2 font-medium text-neutral-900">
          Enter
        </button>
      </form>
    </main>
  );
}
