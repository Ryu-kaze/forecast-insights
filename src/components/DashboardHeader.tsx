import { Download, RefreshCw, Calendar } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { SalesRecord, downloadCSV } from '@/data/salesData';

interface DashboardHeaderProps {
  data: SalesRecord[];
  onRefresh: () => void;
}

const DashboardHeader = ({ data, onRefresh }: DashboardHeaderProps) => {
  const handleExport = () => {
    downloadCSV(data, `sales_forecast_${new Date().toISOString().split('T')[0]}.csv`);
  };

  return (
    <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8 opacity-0 animate-fade-in">
      <div>
        <div className="flex items-center gap-3 mb-2">
          <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
          <span className="text-xs font-medium text-primary uppercase tracking-wider">Live Data</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-bold text-foreground">
          Predictive Sales Dashboard
        </h1>
        <p className="text-muted-foreground mt-1">
          ML-powered forecasting • Python Scikit-learn • Power BI Ready
        </p>
      </div>
      
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-secondary/50 border border-border">
          <Calendar className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm text-muted-foreground">Jan 2024 - Jun 2025</span>
        </div>
        
        <Button
          variant="outline"
          size="icon"
          onClick={onRefresh}
          className="border-border hover:border-primary/50 hover:bg-primary/10"
        >
          <RefreshCw className="w-4 h-4" />
        </Button>
        
        <Button
          onClick={handleExport}
          className="bg-primary hover:bg-primary/90 text-primary-foreground gap-2"
        >
          <Download className="w-4 h-4" />
          Export to Power BI
        </Button>
      </div>
    </header>
  );
};

export default DashboardHeader;
