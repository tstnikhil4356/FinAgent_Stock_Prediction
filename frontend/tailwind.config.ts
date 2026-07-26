import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0b0e1a",
        navy: "#131829",
        navy2: "#1b2137",
        edge: "rgba(255,255,255,0.07)",
        accent: "#5b6ef5",
        accent2: "#8b9bff",
        buy: "#22c55e",
        sell: "#f0554d",
        hold: "#f0b429",
        saffron: "#ff7a1a",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui"],
        display: ["Space Grotesk", "Inter", "ui-sans-serif", "system-ui"],
        mono: ["JetBrains Mono", "ui-monospace", "SF Mono", "monospace"],
      },
      boxShadow: {
        glow: "0 0 40px rgba(91,110,245,0.28)",
      },
    },
  },
  plugins: [],
};
export default config;
