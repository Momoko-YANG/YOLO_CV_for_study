import { useState } from "react";
import LoginPage from "./pages/LoginPage";
import MainPage from "./pages/MainPage";
import type { UserInfo } from "./types";

export default function App() {
  const [user, setUser] = useState<UserInfo | null>(null);

  if (!user) {
    return <LoginPage onLoginSuccess={setUser} />;
  }

  return <MainPage user={user} onLogout={() => setUser(null)} />;
}
