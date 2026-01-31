import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart
} from 'recharts';
import { SalesRecord } from '@/data/salesData';

interface SalesChartProps {
  data: SalesRecord[];
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const actual = payload.find((p: any) => p.dataKey === 'actual');
    const predicted = payload.find((p: any) => p.dataKey === 'predicted');
    const variance = actual && predicted ? actual.value - predicted.value : 0;
    const variancePercent = predicted ? ((variance / predicted.value) * 100).toFixed(2) : 0;

    return (
      <div className="analytics-card !p-4 border border-border/50">
        <p className="text-sm font-medium text-foreground mb-2">{label}</p>
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-primary" />
            <span className="text-sm text-muted-foreground">Actual:</span>
            <span className="text-sm font-medium text-foreground">
              ${actual?.value?.toLocaleString()}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-accent" />
            <span className="text-sm text-muted-foreground">Predicted:</span>
            <span className="text-sm font-medium text-foreground">
              ${predicted?.value?.toLocaleString()}
            </span>
          </div>
          <div className="pt-2 border-t border-border mt-2">
            <span className={`text-sm font-medium ${variance >= 0 ? 'variance-positive' : 'variance-negative'}`}>
              Variance: {variance >= 0 ? '+' : ''}{variancePercent}%
            </span>
          </div>
        </div>
      </div>
    );
  }
  return null;
};

const SalesChart = ({ data }: SalesChartProps) => {
  return (
    <div className="chart-container opacity-0 animate-fade-in" style={{ animationDelay: '200ms' }}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-foreground">Sales Forecast Analysis</h3>
          <p className="text-sm text-muted-foreground">Actual vs ML-Predicted Sales (Scikit-learn)</p>
        </div>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-primary" />
            <span className="text-sm text-muted-foreground">Actual Sales</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-accent" />
            <span className="text-sm text-muted-foreground">Predicted Sales</span>
          </div>
        </div>
      </div>
      
      <div className="h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="actualGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(173, 80%, 45%)" stopOpacity={0.3} />
                <stop offset="95%" stopColor="hsl(173, 80%, 45%)" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="predictedGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(38, 92%, 55%)" stopOpacity={0.2} />
                <stop offset="95%" stopColor="hsl(38, 92%, 55%)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(222, 47%, 18%)" vertical={false} />
            <XAxis 
              dataKey="month" 
              tick={{ fill: 'hsl(215, 20%, 55%)', fontSize: 12 }}
              axisLine={{ stroke: 'hsl(222, 47%, 18%)' }}
              tickLine={false}
            />
            <YAxis 
              tick={{ fill: 'hsl(215, 20%, 55%)', fontSize: 12 }}
              axisLine={{ stroke: 'hsl(222, 47%, 18%)' }}
              tickLine={false}
              tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="actual"
              fill="url(#actualGradient)"
              stroke="transparent"
            />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="hsl(173, 80%, 45%)"
              strokeWidth={3}
              dot={{ fill: 'hsl(173, 80%, 45%)', strokeWidth: 0, r: 4 }}
              activeDot={{ r: 6, fill: 'hsl(173, 80%, 45%)', stroke: 'hsl(173, 80%, 60%)', strokeWidth: 2 }}
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="hsl(38, 92%, 55%)"
              strokeWidth={3}
              strokeDasharray="8 4"
              dot={{ fill: 'hsl(38, 92%, 55%)', strokeWidth: 0, r: 4 }}
              activeDot={{ r: 6, fill: 'hsl(38, 92%, 55%)', stroke: 'hsl(38, 92%, 65%)', strokeWidth: 2 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default SalesChart;
