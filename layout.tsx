import './globals.css';

export const metadata = {
  title: 'Crime Investigation Prototype',
  description: 'Case management and evidence analysis workspace'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
