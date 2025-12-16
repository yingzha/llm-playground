import React from 'react';
import { Gift, PawPrint } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="bg-xmas-red text-white py-6 px-4 shadow-lg border-b-4 border-xmas-gold">
      <div className="max-w-4xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-white p-2 rounded-full text-xmas-red">
            <PawPrint size={32} />
          </div>
          <div>
            <h1 className="text-2xl md:text-3xl font-serif font-bold tracking-wide">Paw-rfect Christmas</h1>
            <p className="text-xmas-gold text-sm font-medium opacity-90">Pet Holiday Transformer</p>
          </div>
        </div>
        <div className="hidden sm:block">
          <Gift size={32} className="text-xmas-gold animate-bounce" />
        </div>
      </div>
    </header>
  );
};

export default Header;