import { SalesRecord } from '@/data/salesData';
import { ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';

interface VarianceTableProps {
  data: SalesRecord[];
}

const VarianceTable = ({ data }: VarianceTableProps) => {
  return (
    <div className="chart-container opacity-0 animate-fade-in" style={{ animationDelay: '300ms' }}>
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-foreground">Variance Analysis</h3>
        <p className="text-sm text-muted-foreground">Monthly breakdown with DAX-calculated metrics</p>
      </div>
      
      <div className="overflow-x-auto">
        <table className="data-table">
          <thead>
            <tr>
              <th>Period</th>
              <th className="text-right">Actual</th>
              <th className="text-right">Predicted</th>
              <th className="text-right">Variance</th>
              <th className="text-right">Variance %</th>
              <th className="text-center">Status</th>
            </tr>
          </thead>
          <tbody>
            {data.map((record, index) => (
              <tr 
                key={record.month}
                className="opacity-0 animate-fade-in"
                style={{ animationDelay: `${400 + index * 50}ms` }}
              >
                <td className="font-medium text-foreground">{record.month}</td>
                <td className="text-right font-mono text-foreground">
                  ${record.actual.toLocaleString()}
                </td>
                <td className="text-right font-mono text-muted-foreground">
                  ${record.predicted.toLocaleString()}
                </td>
                <td className={`text-right font-mono ${record.variance >= 0 ? 'variance-positive' : 'variance-negative'}`}>
                  {record.variance >= 0 ? '+' : ''}${record.variance.toLocaleString()}
                </td>
                <td className={`text-right font-mono ${record.variancePercent >= 0 ? 'variance-positive' : 'variance-negative'}`}>
                  {record.variancePercent >= 0 ? '+' : ''}{record.variancePercent.toFixed(2)}%
                </td>
                <td className="text-center">
                  {record.variance > 0 ? (
                    <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-xs">
                      <ArrowUpRight className="w-3 h-3" />
                      Above
                    </span>
                  ) : record.variance < 0 ? (
                    <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-red-500/10 text-red-400 text-xs">
                      <ArrowDownRight className="w-3 h-3" />
                      Below
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-muted text-muted-foreground text-xs">
                      <Minus className="w-3 h-3" />
                      On Target
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default VarianceTable;
