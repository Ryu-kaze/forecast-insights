import { useState, useEffect } from 'react';
import { DollarSign, TrendingUp, Target, BarChart3, Percent } from 'lucide-react';
import { generateSalesData, calculateKPIs, SalesRecord, KPIData } from '@/data/salesData';
import KPICard from '@/components/KPICard';
import SalesChart from '@/components/SalesChart';
import VarianceTable from '@/components/VarianceTable';
import DAXReference from '@/components/DAXReference';
import DashboardHeader from '@/components/DashboardHeader';

const Index = () => {
  const [salesData, setSalesData] = useState<SalesRecord[]>([]);
  const [kpis, setKpis] = useState<KPIData | null>(null);

  const loadData = () => {
    const data = generateSalesData();
    setSalesData(data);
    setKpis(calculateKPIs(data));
  };

  useEffect(() => {
    loadData();
  }, []);

  const formatCurrency = (value: number) => {
    if (value >= 1000000) {
      return `$${(value / 1000000).toFixed(2)}M`;
    }
    return `$${(value / 1000).toFixed(0)}K`;
  };

  if (!kpis) return null;

  return (
    <div className="min-h-screen bg-background p-4 sm:p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        <DashboardHeader data={salesData} onRefresh={loadData} />

        {/* KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
          <KPICard
            title="Actual Sales"
            value={formatCurrency(kpis.totalActualSales)}
            subtitle="Total revenue"
            icon={DollarSign}
            variant="primary"
            delay={0}
          />
          <KPICard
            title="Predicted Sales"
            value={formatCurrency(kpis.totalPredictedSales)}
            subtitle="ML forecast"
            icon={TrendingUp}
            variant="accent"
            delay={50}
          />
          <KPICard
            title="Variance"
            value={formatCurrency(Math.abs(kpis.overallVariance))}
            subtitle={kpis.overallVariance >= 0 ? 'Above target' : 'Below target'}
            icon={BarChart3}
            trend={kpis.overallVariance >= 0 ? 'up' : 'down'}
            trendValue={`${Math.abs(kpis.overallVariancePercent).toFixed(1)}%`}
            delay={100}
          />
          <KPICard
            title="Forecast Accuracy"
            value={`${kpis.accuracy.toFixed(1)}%`}
            subtitle="Model performance"
            icon={Target}
            delay={150}
          />
          <KPICard
            title="MAPE"
            value={`${kpis.mape.toFixed(2)}%`}
            subtitle="Mean Absolute % Error"
            icon={Percent}
            delay={200}
          />
        </div>

        {/* Main Chart */}
        <div className="mb-8">
          <SalesChart data={salesData} />
        </div>

        {/* Bottom Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <VarianceTable data={salesData} />
          </div>
          <div>
            <DAXReference />
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-12 pt-6 border-t border-border text-center">
          <p className="text-sm text-muted-foreground">
            Data simulated using ML model patterns • Export CSV for Power BI integration
          </p>
        </footer>
      </div>
    </div>
  );
};

export default Index;
