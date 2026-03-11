/**
 * LoginForm — mirrors LoginForm.ui + LoginWindow.py
 * Tabs: Login | Register | Forgot password
 */
import { useState } from "react";
import { authApi } from "../../services/api";
import type { UserInfo } from "../../types";
import "./LoginForm.css";

interface Props {
  onLoginSuccess: (user: UserInfo) => void;
}

type Tab = "login" | "register" | "forgot";

export default function LoginForm({ onLoginSuccess }: Props) {
  const [tab, setTab] = useState<Tab>("login");
  const [info, setInfo] = useState("");

  // Login fields
  const [loginUser, setLoginUser] = useState("");
  const [loginPwd, setLoginPwd] = useState("");

  // Register fields
  const [regUser, setRegUser] = useState("");
  const [regPwd, setRegPwd] = useState("");
  const [regCaptcha, setRegCaptcha] = useState("");
  const [captchaAnswer] = useState(() => generateCaptcha());
  const [avatar, setAvatar] = useState<File | null>(null);

  // ── handlers ────────────────────────────────────────────────────
  async function handleLogin() {
    if (!loginUser || !loginPwd) { setInfo("Please fill in all fields"); return; }
    try {
      const user = await authApi.login({ username: loginUser, password: loginPwd });
      setInfo("Logging in…");
      onLoginSuccess(user);
    } catch (e: unknown) {
      setInfo(e instanceof Error ? e.message : "Login failed");
    }
  }

  async function handleRegister() {
    if (!regUser || !regPwd || !regCaptcha) { setInfo("Please fill in all fields"); return; }
    if (!avatar) { setInfo("Please select an avatar"); return; }
    try {
      await authApi.register(regUser, regPwd, regCaptcha, captchaAnswer, avatar);
      setInfo("Registered successfully! Please log in.");
      setTab("login");
    } catch (e: unknown) {
      setInfo(e instanceof Error ? e.message : "Registration failed");
    }
  }

  // ── render ──────────────────────────────────────────────────────
  return (
    <div className="login-card">
      <div className="login-tabs">
        <button className={tab === "login" ? "active" : ""} onClick={() => setTab("login")}>Login</button>
        <button className={tab === "register" ? "active" : ""} onClick={() => setTab("register")}>Register</button>
        <button className={tab === "forgot" ? "active" : ""} onClick={() => setTab("forgot")}>Forgot</button>
      </div>

      {tab === "login" && (
        <div className="login-fields">
          <input placeholder="Username" maxLength={10} value={loginUser} onChange={e => setLoginUser(e.target.value)} />
          <input placeholder="Password" type="password" maxLength={12} value={loginPwd} onChange={e => setLoginPwd(e.target.value)} />
          <button className="btn-primary" onClick={handleLogin}>Login</button>
        </div>
      )}

      {tab === "register" && (
        <div className="login-fields">
          <input placeholder="Username" maxLength={10} value={regUser} onChange={e => setRegUser(e.target.value)} />
          <input placeholder="Password" type="password" maxLength={12} value={regPwd} onChange={e => setRegPwd(e.target.value)} />
          <div className="captcha-row">
            <input placeholder="Captcha" value={regCaptcha} onChange={e => setRegCaptcha(e.target.value)} />
            <span className="captcha-display">{captchaAnswer}</span>
          </div>
          <label className="avatar-pick">
            {avatar ? avatar.name : "Select avatar"}
            <input type="file" accept="image/*" hidden onChange={e => setAvatar(e.target.files?.[0] ?? null)} />
          </label>
          <button className="btn-primary" onClick={handleRegister}>Register</button>
        </div>
      )}

      {tab === "forgot" && (
        <ForgotPassword onDone={() => setTab("login")} />
      )}

      {info && <p className="login-info">{info}</p>}
    </div>
  );
}

// ── sub-component ────────────────────────────────────────────────────────
function ForgotPassword({ onDone }: { onDone: () => void }) {
  const [user, setUser] = useState("");
  const [pwd, setPwd] = useState("");
  const [captcha, setCaptcha] = useState("");
  const [answer] = useState(generateCaptcha);
  const [msg, setMsg] = useState("");

  async function handle() {
    if (!user || !pwd || !captcha) { setMsg("Fill in all fields"); return; }
    if (captcha.toLowerCase() !== answer.toLowerCase()) { setMsg("Wrong captcha"); return; }
    try {
      await authApi.changePassword(user, pwd, captcha);
      setMsg("Password changed!");
      setTimeout(onDone, 1500);
    } catch (e: unknown) {
      setMsg(e instanceof Error ? e.message : "Failed");
    }
  }

  return (
    <div className="login-fields">
      <input placeholder="Username" value={user} onChange={e => setUser(e.target.value)} />
      <input placeholder="New password" type="password" value={pwd} onChange={e => setPwd(e.target.value)} />
      <div className="captcha-row">
        <input placeholder="Captcha" value={captcha} onChange={e => setCaptcha(e.target.value)} />
        <span className="captcha-display">{answer}</span>
      </div>
      <button className="btn-primary" onClick={handle}>Change password</button>
      {msg && <p>{msg}</p>}
    </div>
  );
}

function generateCaptcha(): string {
  return Math.random().toString(36).slice(2, 6).toUpperCase();
}
