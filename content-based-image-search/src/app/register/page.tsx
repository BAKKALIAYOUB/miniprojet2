// src/app/register/page.tsx

'use client';

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function RegisterPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");
  const [error, setError] = useState("");
  const router = useRouter();

const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();

  // Check if all required fields are filled
  if (!username || !password || !email || !phone) {
    setError("All fields are required.");
    return;
  }

  try {
    console.log("Sending data:", { username, password, email, phone });
    const response = await fetch("http://localhost:8000/registeration", {
    method: "POST",
    headers: {
        "Content-Type": "application/json",
    },
    body: JSON.stringify({
        username,
        password,
        email,
        phone,
    }),
});

    // Function to read and handle the response body
    const parseResponseBody = async (response: Response) => {
      const contentType = response.headers.get("Content-Type");

      // If response is JSON, try to parse it
      if (contentType && contentType.includes("application/json")) {
        return response.json();
      }

      // If response is not JSON, fallback to text
      return response.text();
    };

    const data = await parseResponseBody(response);

    console.log("Server response:", data);

    if (!response.ok) {
      setError(data.message || "Something went wrong");
    } else {
      router.push("/login");
    }
  } catch (error) {
    console.error("Registration failed", error);
    setError("An error occurred. Please try again.");
  }
};





  return (
    <div className="flex items-center justify-center h-screen bg-gradient-to-r from-purple-400 via-pink-500 to-red-500">
      <form
        onSubmit={handleSubmit}
        className="w-96 p-8 bg-white shadow-xl rounded-lg"
      >
        <h1 className="text-2xl font-semibold text-center mb-6 text-gray-800">Create Account</h1>
        {error && <p className="text-red-500 mb-4 text-center">{error}</p>}

        {/* Username Field */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2 text-gray-600">Username</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="w-full border-2 border-gray-300 rounded-md p-3 focus:outline-none focus:border-blue-500 transition"
            required
          />
        </div>

        {/* Password Field */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2 text-gray-600">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full border-2 border-gray-300 rounded-md p-3 focus:outline-none focus:border-blue-500 transition"
            required
          />
        </div>

        {/* Email Field */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2 text-gray-600">Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full border-2 border-gray-300 rounded-md p-3 focus:outline-none focus:border-blue-500 transition"
            required
          />
        </div>

        {/* Phone Number Field */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2 text-gray-600">Phone Number</label>
          <input
            type="text"
            value={phone}
            onChange={(e) => setPhone(e.target.value)}
            className="w-full border-2 border-gray-300 rounded-md p-3 focus:outline-none focus:border-blue-500 transition"
            required
          />
        </div>

        <button
          type="submit"
          className="w-full bg-gradient-to-r from-blue-500 to-purple-500 text-white py-3 rounded-md mt-4 hover:from-blue-600 hover:to-purple-600 transition"
        >
          Register
        </button>

        <div className="mt-6 text-center">
          <p className="text-sm text-gray-600">
            Already have an account?{" "}
            <a href="/login" className="text-blue-500 hover:underline">Login</a>
          </p>
        </div>
      </form>
    </div>
  );
}
