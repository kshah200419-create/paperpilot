import clsx from "clsx";

export function CategoryBadge({ category, primary }: { category: string; primary?: boolean }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-full px-2 py-0.5 font-mono text-[11px] font-medium",
        primary ? "bg-indigo-100 text-indigo-700" : "bg-slate-100 text-slate-600",
      )}
    >
      {category}
    </span>
  );
}

export function TopicBadge({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center rounded-full bg-emerald-50 px-2 py-0.5 text-[11px] font-medium text-emerald-700 ring-1 ring-emerald-200">
      {label}
    </span>
  );
}
