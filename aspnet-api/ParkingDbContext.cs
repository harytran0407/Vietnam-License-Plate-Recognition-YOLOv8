using Microsoft.EntityFrameworkCore;
using ParkingSystem.Models;

namespace ParkingSystem.Data;

public class ParkingDbContext(DbContextOptions<ParkingDbContext> options)
    : DbContext(options)
{
    public DbSet<ParkingRecord> ParkingRecords => Set<ParkingRecord>();

    protected override void OnModelCreating(ModelBuilder mb)
    {
        mb.Entity<ParkingRecord>(e =>
        {
            e.HasIndex(r => r.LicensePlate);
            e.HasIndex(r => r.Status);
            e.HasIndex(r => r.CheckInTime);
            // Fee is computed – not stored in DB
            e.Ignore(r => r.Fee);
        });
    }
}
