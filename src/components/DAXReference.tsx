import { daxFormulas } from '@/data/salesData';
import { Code, Copy, Check } from 'lucide-react';
import { useState } from 'react';

const DAXReference = () => {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const formulas = Object.entries(daxFormulas).map(([key, formula]) => ({
    name: key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase()),
    formula
  }));

  const copyToClipboard = (formula: string, index: number) => {
    navigator.clipboard.writeText(formula);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div className="chart-container opacity-0 animate-fade-in" style={{ animationDelay: '400ms' }}>
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-lg bg-accent/10">
          <Code className="w-5 h-5 text-accent" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-foreground">Power BI DAX Formulas</h3>
          <p className="text-sm text-muted-foreground">Ready-to-use measures for variance analysis</p>
        </div>
      </div>
      
      <div className="space-y-3">
        {formulas.map((item, index) => (
          <div 
            key={item.name}
            className="group flex items-center justify-between p-3 rounded-lg bg-secondary/30 border border-border/50 hover:border-accent/30 transition-colors"
          >
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-accent mb-1">{item.name}</p>
              <code className="text-sm text-muted-foreground font-mono block truncate">
                {item.formula}
              </code>
            </div>
            <button
              onClick={() => copyToClipboard(item.formula, index)}
              className="ml-3 p-2 rounded-md bg-secondary/50 hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors opacity-0 group-hover:opacity-100"
            >
              {copiedIndex === index ? (
                <Check className="w-4 h-4 text-emerald-400" />
              ) : (
                <Copy className="w-4 h-4" />
              )}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DAXReference;
