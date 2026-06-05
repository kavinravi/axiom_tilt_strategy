"use client";
export function WeekPicker({ fridays, selected, onPick }: {
  fridays: string[]; selected: string | null; onPick: (f: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {fridays.map((f) => (
        <button key={f} onClick={() => onPick(f)}
          className={`rounded-md px-2.5 py-1 text-sm ring-1 ${
            f === selected ? "bg-neutral-100 text-neutral-900 ring-neutral-100" : "bg-neutral-900 text-neutral-300 ring-neutral-700"
          }`}>
          {f}
        </button>
      ))}
    </div>
  );
}
