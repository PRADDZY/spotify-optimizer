import "./globals.css";
import { Alegreya_Sans, Cinzel } from "next/font/google";

const display = Cinzel({
  subsets: ["latin"],
  weight: ["400", "700"],
  variable: "--font-display",
});

const body = Alegreya_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  variable: "--font-body",
});

export const metadata = {
  title: "Mix Optimizer",
  description: "Build smoother Spotify playlists with BPM and key-aware ordering.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable}`}>
      <body>
        <div className="noise" aria-hidden="true" />
        {children}
      </body>
    </html>
  );
}
