import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";

export const authOptions = {
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        username: { label: "Username", type: "text" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials: { username: string; password: string } | undefined) {
        if (!credentials) return null;

        const { username, password } = credentials;

        // Exemple d'authentification (remplacez par votre logique)
        if (username === "admin" && password === "admin") {
          return { id: 1, name: "Admin", email: "admin@example.com" }; // Remplacez par vos propres données utilisateur
        }

        return null; // Retourner null si l'authentification échoue
      },
    }),
  ],
  pages: {
    signIn: "/login", // Redirige vers la page de login
    signOut: "/login",     // Redirige vers la page d'accueil après déconnexion
  },
  session: {
    strategy: "jwt",  // Utilise JWT pour la gestion des sessions
  },
};

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
