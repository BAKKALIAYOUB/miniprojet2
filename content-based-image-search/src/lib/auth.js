import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";

export default NextAuth({
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        username: { label: "Username", type: "text" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        const { username, password } = credentials;

        // Exemple simple : Remplacez par un appel API ou une base de données
        if (username === "admin" && password === "admin") {
          return { id: 1, name: "Admin", email: "admin@example.com" };
        }
        return null; // Retourne null pour rejeter l'accès
      },
    }),
  ],
  pages: {
    signIn: "/login", // Redirige les utilisateurs vers votre page de connexion
  },
  session: {
    jwt: true, // Utilisation des sessions JWT
  },
});
