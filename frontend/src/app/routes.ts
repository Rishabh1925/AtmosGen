import { createBrowserRouter } from "react-router";
import { AppLayout } from "./layouts/AppLayout";
import { LandingPage } from "./pages/LandingPage";
import { HomePage } from "./pages/HomePage";
import { LoginPage } from "./pages/LoginPage";
import { RegisterPage } from "./pages/RegisterPage";
import { DashboardPage } from "./pages/DashboardPage";
import { ForecastPage } from "./pages/ForecastPage";
import { ForecastDetailPage } from "./pages/ForecastDetailPage";
import { ForecastHistoryPage } from "./pages/ForecastHistoryPage";
import { SatellitePage } from "./pages/SatellitePage";
import { ContactPage } from "./pages/ContactPage";
import { NotFoundPage } from "./pages/NotFoundPage";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: AppLayout,
    children: [
      {
        index: true,
        Component: LandingPage,
      },
      {
        path: "home",
        Component: HomePage,
      },
      {
        path: "login",
        Component: LoginPage,
      },
      {
        path: "register",
        Component: RegisterPage,
      },
      {
        path: "dashboard",
        Component: DashboardPage,
      },
      {
        path: "forecast",
        Component: ForecastPage,
      },
      {
        path: "forecast/:id",
        Component: ForecastDetailPage,
      },
      {
        path: "history",
        Component: ForecastHistoryPage,
      },
      {
        path: "satellite",
        Component: SatellitePage,
      },
      {
        path: "contact",
        Component: ContactPage,
      },
      {
        path: "*",
        Component: NotFoundPage,
      },
    ],
  },
]);
