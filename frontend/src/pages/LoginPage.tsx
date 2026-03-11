import LoginForm from "../components/Login/LoginForm";
import type { UserInfo } from "../types";

interface Props {
  onLoginSuccess: (user: UserInfo) => void;
}

export default function LoginPage({ onLoginSuccess }: Props) {
  return (
    <div style={{ minHeight: "100vh", background: "#f0f4ff", display: "flex", alignItems: "center" }}>
      <LoginForm onLoginSuccess={onLoginSuccess} />
    </div>
  );
}
