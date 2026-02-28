import { Outlet } from "react-router";

export function RootLayout() {
  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-blue-50 via-sky-50 to-indigo-100 text-slate-800 font-sans selection:bg-blue-200">
      <Outlet />
    </div>
  );
}
