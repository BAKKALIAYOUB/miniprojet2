// src/app/login/page.tsx

'use client';

import { signIn } from "next-auth/react";
import { useState } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();

  // Check if both fields are filled
  if (!username || !password) {
    setError("Both fields are required.");
    return;
  }

  try {
    const response = await fetch("http://localhost:8000/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({
        username: username,
        password: password,
      }).toString(),
    });

    const data = await response.json();

    if (response.ok) {
      // Save the JWT token to sessionStorage
      sessionStorage.setItem("jwtToken", data.token);

      console.log("hi",data.token);
      router.push("/gestion_images"); // Redirect on successful login
    } else {
      setError(data.detail || "Invalid username or password");
    }
  } catch (error) {
    console.error("Login failed", error);
    setError("An error occurred. Please try again.");
  }
};

  return (
    <div className="flex items-center justify-center h-screen">
      <form
        onSubmit={handleSubmit}
        className="w-96 p-6 bg-white shadow-md rounded-md"
      >
        <h1 className="text-xl font-bold mb-4">Login</h1>
        {error && <p className="text-red-500 mb-4">{error}</p>}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Username</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="w-full border rounded-md p-2"
            required
          />
        </div>
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full border rounded-md p-2"
            required
          />
        </div>
        <button
          type="submit"
          className="w-full bg-blue-500 text-white py-2 rounded-md"
        >
          Login
        </button>
        <div className="mt-4">
          <p className="text-sm text-center">
            Don't have an account?{" "}
            <a href="/register" className="text-blue-500">Register</a>
          </p>
        </div>
      </form>
    </div>
  );
}
