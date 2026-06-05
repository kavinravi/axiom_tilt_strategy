"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { href: "/now", label: "Now" },
  { href: "/holdings", label: "Holdings" },
  { href: "/history", label: "History" },
];

export default function DashLayout({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  return (
    <div className="mx-auto max-w-3xl px-4 pb-16">
      <header className="flex items-center justify-between py-4">
        <span className="font-semibold">Axiom Tilt</span>
      </header>
      <nav className="mb-4 flex gap-1 rounded-lg bg-neutral-900 p-1 ring-1 ring-neutral-800">
        {TABS.map((t) => (
          <Link
            key={t.href} href={t.href}
            className={`flex-1 rounded-md px-3 py-2 text-center text-sm ${
              path === t.href ? "bg-neutral-100 text-neutral-900 font-medium" : "text-neutral-300"
            }`}
          >
            {t.label}
          </Link>
        ))}
      </nav>
      {children}
    </div>
  );
}
