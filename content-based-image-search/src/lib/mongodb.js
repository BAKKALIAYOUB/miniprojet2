import { MongoClient } from "mongodb";

// Mongo URI with database name (replace 'your-db-name' with the actual database name)
const client = new MongoClient("mongodb://localhost:27017"); // Mongo URI with protocol

export const connectToDatabase = async () => {
  // Connect to the database
  await client.connect();

  // Use a specific database, replace 'your-db-name' with the actual name of your database
  return client.db("users"); // specify the 'client' database
};
