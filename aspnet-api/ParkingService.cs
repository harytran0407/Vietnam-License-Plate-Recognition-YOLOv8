using Microsoft.EntityFrameworkCore;
using ParkingSystem.Data;
using ParkingSystem.Models;

namespace ParkingSystem.Services;

// ── Interface ─────────────────────────────────────────────────────────────
public interface IParkingService
{
    Task<ParkingEventResponse> CheckInAsync(ParkingEventDto dto);
    Task<ParkingEventResponse> CheckOutAsync(ParkingEventDto dto);
    Task<List<ParkingRecord>>  GetActiveAsync();
    Task<List<ParkingRecord>>  GetHistoryAsync(int page, int pageSize, string? plate);
    Task<ParkingStats>         GetStatsAsync();
    Task<ParkingRecord?>       GetByIdAsync(string id);
}

// ── Implementation ────────────────────────────────────────────────────────
public class ParkingService(ParkingDbContext db) : IParkingService
{
    // ── Check-in ──────────────────────────────────────────────────────
    public async Task<ParkingEventResponse> CheckInAsync(ParkingEventDto dto)
    {
        // Idempotency: ignore if record already exists
        if (await db.ParkingRecords.AnyAsync(r => r.Id == dto.Id))
            return new(true, "Already exists", dto.Id);

        var record = new ParkingRecord
        {
            Id           = dto.Id,
            LicensePlate = dto.LicensePlate.ToUpper().Trim(),
            CameraId     = dto.CameraId,
            CheckInTime  = ParseDt(dto.CheckinTime) ?? DateTime.Now,
            Status       = "PARKING",
            SnapshotPath = dto.SnapshotPath,
        };

        db.ParkingRecords.Add(record);
        await db.SaveChangesAsync();
        return new(true, "Check-in recorded", record.Id);
    }

    // ── Check-out ─────────────────────────────────────────────────────
    public async Task<ParkingEventResponse> CheckOutAsync(ParkingEventDto dto)
    {
        // Find the PARKING record for this plate
        var record = await db.ParkingRecords
            .Where(r => r.LicensePlate == dto.LicensePlate.ToUpper().Trim()
                     && r.Status == "PARKING")
            .OrderByDescending(r => r.CheckInTime)
            .FirstOrDefaultAsync();

        if (record is null)
            return new(false, "No active parking record found", string.Empty);

        record.CheckOutTime  = ParseDt(dto.CheckoutTime) ?? DateTime.Now;
        record.TotalMinutes  = dto.TotalMinutes;
        record.TotalHours    = dto.TotalHours;
        record.Status        = "CHECKED_OUT";

        await db.SaveChangesAsync();
        return new(true, $"Check-out OK. Fee: {record.Fee:N0} VND", record.Id);
    }

    // ── Queries ───────────────────────────────────────────────────────
    public Task<List<ParkingRecord>> GetActiveAsync() =>
        db.ParkingRecords
          .Where(r => r.Status == "PARKING")
          .OrderByDescending(r => r.CheckInTime)
          .ToListAsync();

    public Task<List<ParkingRecord>> GetHistoryAsync(
        int page, int pageSize, string? plate)
    {
        var q = db.ParkingRecords.AsQueryable();
        if (!string.IsNullOrWhiteSpace(plate))
            q = q.Where(r => r.LicensePlate.Contains(plate.ToUpper()));
        return q.OrderByDescending(r => r.CheckInTime)
                .Skip((page - 1) * pageSize)
                .Take(pageSize)
                .ToListAsync();
    }

    public Task<ParkingRecord?> GetByIdAsync(string id) =>
        db.ParkingRecords.FindAsync(id).AsTask();

    public async Task<ParkingStats> GetStatsAsync()
    {
        var today = DateTime.Today;
        var currently = await db.ParkingRecords.CountAsync(r => r.Status == "PARKING");
        var totalToday = await db.ParkingRecords
            .CountAsync(r => r.CheckInTime >= today);
        var totalAll = await db.ParkingRecords.CountAsync();
        var feeToday = await db.ParkingRecords
            .Where(r => r.CheckInTime >= today && r.Status == "CHECKED_OUT")
            .SumAsync(r => (decimal)(r.TotalHours > 0
                ? 5000 + Math.Max(0, Math.Ceiling(r.TotalHours) - 1) * 3000
                : 0));

        return new ParkingStats(currently, totalToday, totalAll, feeToday);
    }

    // ── Helper ────────────────────────────────────────────────────────
    private static DateTime? ParseDt(string s) =>
        DateTime.TryParse(s, out var dt) ? dt : null;
}
