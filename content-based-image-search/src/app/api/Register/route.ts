// /src/app/api/register/route.ts
import { NextResponse } from "next/server";
import { connectToDatabase } from "@/lib/mongodb";
import bcrypt from "bcryptjs"; // For password hashing

export async function POST(request: Request) {
  console.log("hi")
  try {
    const { username, password, email, phone } = await request.json();

    // Log incoming data
    console.log("Received data:", { username, password, email, phone });

    // Ensure all required fields are provided
    if (!username || !password || !email || !phone) {
      return new NextResponse(
        JSON.stringify({ message: "All fields are required." }),
        { status: 400 }
      );
    }

    const db = await connectToDatabase();

    // Check if the username already exists
    const existingUser = await db.collection("users").findOne({ username });
    if (existingUser) {
      return new NextResponse(
        JSON.stringify({ message: "Username already exists" }),
        { status: 400 }
      );
    }

    // Hash the password before saving to the database
    const hashedPassword = await bcrypt.hash(password, 12);

    // Create the new user and save it to the database
    const newUser = await db.collection("users").insertOne({
      username,
      password: hashedPassword,
      email,
      phone,
    });

    return new NextResponse(
      JSON.stringify({ message: "User registered successfully" }),
      { status: 200 }
    );
  } catch (error) {
    console.error("Error processing registration:", error);
    return new NextResponse(
      JSON.stringify({ message: "Internal Server Error" }),
      { status: 500 }
    );
  }
}
