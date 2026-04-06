/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'cyber': {
          50: '#ecfeff',
          100: '#cffafe',
          200: '#a5f3fc',
          300: '#67e8f9',
          400: '#22d3ee',
          500: '#06b6d4',
          600: '#0891b2',
          700: '#0e7490',
          800: '#155e75',
          900: '#164e63',
          950: '#083344',
        },
        'surface': {
          DEFAULT: '#111318',
          raised: '#16181d',
          overlay: '#1c1f26',
          elevated: '#22252e',
        },
        'border': {
          DEFAULT: '#2a2d37',
          subtle: '#22252e',
          strong: '#3a3d47',
        },
      },
      fontFamily: {
        'mono': ['JetBrains Mono', 'Fira Code', 'ui-monospace', 'monospace'],
        'sans': ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      fontSize: {
        '2xs': ['0.625rem', { lineHeight: '0.875rem' }],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'slide-up': 'slideUp 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
        'slide-down': 'slideDown 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
        'fade-in': 'fadeIn 0.15s ease-out',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 6px rgb(6 182 212 / 0.2), 0 0 12px rgb(6 182 212 / 0.1)' },
          '100%': { boxShadow: '0 0 12px rgb(6 182 212 / 0.35), 0 0 24px rgb(6 182 212 / 0.15)' },
        },
        slideUp: {
          '0%': { transform: 'translateY(8px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-8px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
      },
      backgroundImage: {
        'grid-pattern': `linear-gradient(to right, rgb(42 45 55 / 0.4) 1px, transparent 1px),
                         linear-gradient(to bottom, rgb(42 45 55 / 0.4) 1px, transparent 1px)`,
        'radial-glow': 'radial-gradient(ellipse at top, rgb(6 182 212 / 0.06) 0%, transparent 60%)',
      },
      backgroundSize: {
        'grid': '24px 24px',
      },
      borderRadius: {
        'xl': '0.75rem',
        '2xl': '1rem',
      },
      boxShadow: {
        'card': '0 1px 3px 0 rgb(0 0 0 / 0.3), 0 1px 2px -1px rgb(0 0 0 / 0.3)',
        'card-hover': '0 4px 16px -2px rgb(0 0 0 / 0.4), 0 2px 6px -2px rgb(0 0 0 / 0.3)',
        'modal': '0 24px 64px -16px rgb(0 0 0 / 0.6), 0 8px 24px -8px rgb(0 0 0 / 0.4)',
        'glow-sm': '0 0 8px -2px rgb(6 182 212 / 0.25)',
        'glow-md': '0 0 16px -4px rgb(6 182 212 / 0.3)',
      },
    },
  },
  plugins: [],
}
