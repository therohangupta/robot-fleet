/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Mission Control Theme
        'cyber': {
          50: '#e6feff',
          100: '#b3fcff',
          200: '#80faff',
          300: '#4df8ff',
          400: '#1af6ff',
          500: '#00d9ff', // Primary cyan
          600: '#00ade6',
          700: '#0082b3',
          800: '#005780',
          900: '#002b4d',
        },
        'slate': {
          850: '#1a1f2e',
          950: '#0d1117',
        }
      },
      fontFamily: {
        'mono': ['JetBrains Mono', 'Fira Code', 'Monaco', 'monospace'],
        'sans': ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgb(0 217 255 / 0.3), 0 0 10px rgb(0 217 255 / 0.2)' },
          '100%': { boxShadow: '0 0 10px rgb(0 217 255 / 0.5), 0 0 20px rgb(0 217 255 / 0.3)' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
      backgroundImage: {
        'grid-pattern': `linear-gradient(to right, rgb(30 41 59 / 0.3) 1px, transparent 1px),
                         linear-gradient(to bottom, rgb(30 41 59 / 0.3) 1px, transparent 1px)`,
        'radial-glow': 'radial-gradient(ellipse at center, rgb(0 217 255 / 0.1) 0%, transparent 70%)',
      },
      backgroundSize: {
        'grid': '24px 24px',
      },
    },
  },
  plugins: [],
}
