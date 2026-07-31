# Nimbus Cloud — Data Retention, Backups, and Export

Nimbus takes automated backups every six hours and keeps them for 30 days on all plans. Enterprise customers can extend backup retention to 90 days on request.

When a customer deletes a record, it is soft-deleted and recoverable for 30 days before it is permanently purged. Purged data cannot be recovered.

If an account is cancelled, data is kept in a read-only state for 60 days so the customer can still export it, after which the account and all its data are permanently deleted.

Customers can export all of their data at any time as newline-delimited JSON through the export API or the account dashboard. There is no charge for exports.
