using Microsoft.EntityFrameworkCore;
using ParkingSystem.Data;
using ParkingSystem.Services;

var builder = WebApplication.CreateBuilder(args);

// ══════════════════════════════════════════════════════════════════
// Services
// ══════════════════════════════════════════════════════════════════
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// ── Database: MySQL (Pomelo) ──────────────────────────────────────
var connStr = builder.Configuration.GetConnectionString("DefaultConnection")
              ?? throw new InvalidOperationException(
                     "Connection string 'DefaultConnection' is missing in appsettings.json");

builder.Services.AddDbContext<ParkingDbContext>(opt =>
    opt.UseMySql(connStr, ServerVersion.AutoDetect(connStr),
        mySqlOpt => mySqlOpt.EnableRetryOnFailure(
            maxRetryCount: 5,
            maxRetryDelay: TimeSpan.FromSeconds(10),
            errorNumbersToAdd: null)));

Console.WriteLine("[DB] Using MySQL");

builder.Services.AddScoped<IParkingService, ParkingService>();

// ── CORS: allow ALL origins (Python AI + any browser client) ─────
// In production, replace AllowAnyOrigin with specific origins.
builder.Services.AddCors(options =>
    options.AddPolicy("AllowAll", policy =>
        policy.AllowAnyOrigin()
              .AllowAnyHeader()
              .AllowAnyMethod()));

// ══════════════════════════════════════════════════════════════════
// Middleware pipeline
// ══════════════════════════════════════════════════════════════════
var app = builder.Build();

// Auto-create DB schema on startup
using (var scope = app.Services.CreateScope())
{
    try
    {
        var db = scope.ServiceProvider.GetRequiredService<ParkingDbContext>();
        db.Database.EnsureCreated();
        Console.WriteLine("[DB] Schema ready.");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[DB ERROR] {ex.Message}");
        Console.WriteLine("[DB] Check your connection string in appsettings.json");
    }
}

app.UseSwagger();
app.UseSwaggerUI(c =>
{
    c.SwaggerEndpoint("/swagger/v1/swagger.json", "Parking API v1");
    c.RoutePrefix = "swagger";
});

app.UseCors("AllowAll");
app.UseDefaultFiles();   // serves wwwroot/index.html for "/"
app.UseStaticFiles();    // serves wwwroot/ assets

// ── Health check endpoint ─────────────────────────────────────────
app.MapGet("/health", () => Results.Ok(new
{
    status  = "ok",
    time    = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
    db      = "mysql",
}));

app.UseAuthorization();
app.MapControllers();

// ── SPA fallback: unknown routes → index.html ────────────────────
app.MapFallbackToFile("index.html");

Console.WriteLine("==============================================");
Console.WriteLine($"  Parking System running at: {string.Join(", ", builder.WebHost.GetSetting("urls") ?? "http://localhost:5000")}");
Console.WriteLine("  Dashboard : http://localhost:5000");
Console.WriteLine("  Swagger   : http://localhost:5000/swagger");
Console.WriteLine("  Health    : http://localhost:5000/health");
Console.WriteLine("==============================================");

app.Run();
