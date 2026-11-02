# Budget Uploads and Versions

Budgets are uploaded to the planning workspace as CSV files. Each upload creates a new budget version; earlier versions are retained for audit.

The CSV format requires three columns: account code, period (YYYY-MM), and amount. Amounts are signed: negative values reduce budget.

To upload a budget:
1. Open Planning, select Budgets, then Upload.
2. Choose the scenario (Base, Stretch, or Forecast) and the CSV file.
3. The system validates every account code and period before committing. Invalid rows are rejected with a row-level report.
4. Confirm to create the new budget version.

Versions can be compared in the Variance view, which shows the delta between any two versions by account and period.
