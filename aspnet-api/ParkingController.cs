using Microsoft.AspNetCore.Mvc;
using ParkingSystem.Models;
using ParkingSystem.Services;

namespace ParkingSystem.Controllers;

/// <summary>
/// REST API for the AI Parking System.
///
/// Python AI service calls:
///   POST /api/parking/checkin
///   POST /api/parking/checkout
///
/// Dashboard / management calls:
///   GET  /api/parking/active
///   GET  /api/parking/history
///   GET  /api/parking/stats
///   GET  /api/parking/{id}
/// </summary>
[ApiController]
[Route("api/[controller]")]
[Produces("application/json")]
public class ParkingController(IParkingService svc) : ControllerBase
{
    // ── Python AI → ASP.NET ───────────────────────────────────────────

    /// <summary>Called by Python when a new vehicle is detected (check-in).</summary>
    [HttpPost("checkin")]
    [ProducesResponseType(typeof(ParkingEventResponse), 200)]
    public async Task<IActionResult> CheckIn([FromBody] ParkingEventDto dto)
    {
        if (string.IsNullOrWhiteSpace(dto.LicensePlate))
            return BadRequest(new ParkingEventResponse(false, "License plate is required", string.Empty));

        var result = await svc.CheckInAsync(dto);
        return result.Success ? Ok(result) : Conflict(result);
    }

    /// <summary>Called by Python when the same vehicle is detected again after 30 s (check-out).</summary>
    [HttpPost("checkout")]
    [ProducesResponseType(typeof(ParkingEventResponse), 200)]
    public async Task<IActionResult> CheckOut([FromBody] ParkingEventDto dto)
    {
        if (string.IsNullOrWhiteSpace(dto.LicensePlate))
            return BadRequest(new ParkingEventResponse(false, "License plate is required", string.Empty));

        var result = await svc.CheckOutAsync(dto);
        return result.Success ? Ok(result) : NotFound(result);
    }

    // ── Dashboard / Management ────────────────────────────────────────

    /// <summary>Returns all vehicles currently inside the parking lot.</summary>
    [HttpGet("active")]
    public async Task<IActionResult> GetActive() =>
        Ok(await svc.GetActiveAsync());

    /// <summary>
    /// Paginated history. Optionally filter by partial plate number.
    /// GET /api/parking/history?page=1&pageSize=20&plate=51F
    /// </summary>
    [HttpGet("history")]
    public async Task<IActionResult> GetHistory(
        [FromQuery] int    page     = 1,
        [FromQuery] int    pageSize = 20,
        [FromQuery] string? plate   = null)
    {
        page     = Math.Max(1, page);
        pageSize = Math.Clamp(pageSize, 1, 100);
        return Ok(await svc.GetHistoryAsync(page, pageSize, plate));
    }

    /// <summary>Dashboard summary statistics.</summary>
    [HttpGet("stats")]
    public async Task<IActionResult> GetStats() =>
        Ok(await svc.GetStatsAsync());

    /// <summary>Get a single record by ID.</summary>
    [HttpGet("{id}")]
    public async Task<IActionResult> GetById(string id)
    {
        var rec = await svc.GetByIdAsync(id);
        return rec is not null ? Ok(rec) : NotFound();
    }
}
