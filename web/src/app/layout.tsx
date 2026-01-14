import "./globals.css";
import type { ReactNode } from "react";
import Providers from "@/app/providers";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import ChatWidget from "@/components/ChatWidget";

export const metadata = {
  title: "AI Stocks Predictor",
  description: "Technical indicators + RandomForest over EODHD data",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-zinc-950 text-zinc-50 antialiased selection:bg-blue-500/30 pt-32">
        <Providers>
          <Header />
          <main className="mx-auto w-full max-w-5xl px-6 pb-12">
            {children}
          </main>
          <Footer />
        </Providers>
      </body>
    </html>
  );
}
