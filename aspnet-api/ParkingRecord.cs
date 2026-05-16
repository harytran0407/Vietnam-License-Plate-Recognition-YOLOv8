using System.ComponentModel.DataAnnotations;

namespace ParkingSystem.Models;

/// <summary>EF Core entity – one row per vehicle visit.</summary>
public class ParkingRecord
{
    [Key]
    public string Id { get; set; } = Guid.NewGuid().ToString()[..8].ToUpper();

    [Required, MaxLength(20)]
    public string LicensePlate { get; set; } = string.Empty;

    [MaxLength(20)]
    public string CameraId { get; set; } = "CAM-01";

    public DateTime CheckInTime  { get; set; } = DateTime.Now;
    public DateTime? CheckOutTime { get; set; }

    public double TotalMinutes { get; set; }
    public double TotalHours   { get; set; }

    /// <summary>PARKING | CHECKED_OUT</summary>
    [MaxLength(20)]
    public string Status { get; set; } = "PARKING";

    [MaxLength(300)]
    public string SnapshotPath { get; set; } = string.Empty;

    // ── Computed helpers ──────────────────────────────────────────────
    /// <summary>Parking fee in VND (first hour 5,000 VND; each extra hour 3,000 VND).</summary>
    public decimal Fee
    {
        get
        {
            if (TotalHours <= 0) return 0;
            decimal first = 5_000m;
            decimal extra = (decimal)Math.Max(0, Math.Ceiling(TotalHours) - 1) * 3_000m;
            return first + extra;
        }
    }
}

// ── DTOs ──────────────────────────────────────────────────────────────────

/// <summary>Payload sent by the Python AI service.</summary>
public class ParkingEventDto
{
    public string Id            { get; set; } = string.Empty;
    public string LicensePlate  { get; set; } = string.Empty;
    public string CameraId      { get; set; } = "CAM-01";
    public string CheckinTime   { get; set; } = string.Empty;
    public string CheckoutTime  { get; set; } = string.Empty;
    public double TotalMinutes  { get; set; }
    public double TotalHours    { get; set; }
    public string Status        { get; set; } = string.Empty;
    public string SnapshotPath  { get; set; } = string.Empty;
}

/// <summary>Response returned to the Python service.</summary>
public record ParkingEventResponse(bool Success, string Message, string RecordId);

/// <summary>Dashboard summary card.</summary>
public record ParkingStats(
    int  CurrentlyParked,
    int  TotalToday,
    int  TotalAllTime,
    decimal TotalFeeToday
);
