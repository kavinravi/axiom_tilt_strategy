export function Empty({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-dashed border-neutral-700 p-8 text-center">
      <p className="font-medium text-neutral-200">{title}</p>
      {hint && <p className="mt-1 text-sm text-neutral-400">{hint}</p>}
    </div>
  );
}
