import { cn } from "../../lib/utils";

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
  hoverEffect?: boolean;
}

export function GlassCard({ children, className, hoverEffect = false, ...props }: GlassCardProps) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-2xl border border-white/20 bg-white/10 backdrop-blur-md shadow-lg transition-all duration-300",
        hoverEffect && "hover:bg-white/20 hover:scale-[1.02] hover:shadow-xl cursor-pointer",
        className
      )}
      {...props}
    >
      <div className="relative z-10">{children}</div>
      {/* Optional: Add a subtle shine or gradient overlay here if needed */}
    </div>
  );
}
