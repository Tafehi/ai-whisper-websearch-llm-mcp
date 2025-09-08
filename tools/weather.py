"""
Open-Meteo Weather Tool Server
Free, open-source weather API.
"""

import os
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()
url = os.getenv("WEATHER_URL")
api_key = os.getenv("WEATHER_API_KEY")

# Validate environment variables
if not url or not api_key:
    raise EnvironmentError("Missing WEATHER_URL or WEATHER_API_KEY in .env file")

# Initialize MCP
mcp = FastMCP(name="weather", host="localhost", port=8002)

@mcp.tool()
async def get_weather(city_name: str) -> dict:
    """
    Fetch current weather for a given city using the API.
    Args:
        city_name (str): Name of the city to fetch weather for.
    Returns:
        dict: Weather details including temperature, wind, country, city, latitude, and longitude and the last updated time.
    """
    print(f"Server received weather request: {city_name}")
    params = {"key": api_key, "q": city_name, "aqi": "no"}
    print(f"Requesting weather data for {city_name} with params: {params}")

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=5)

    if response.status_code == 200:
        data = response.json()
        return {
            "city": data["location"]["name"],
            "country": data["location"]["country"],
            "latitude": data["location"]["lat"],
            "longitude": data["location"]["lon"],
            "last_updated": data["current"]["last_updated"],
            "temperature_c": data["current"]["temp_c"],
            "condition": data["current"]["condition"]["text"],
            "wind_kph": data["current"]["wind_kph"],
        }
    else:
        raise Exception(
            f"API request failed: {response.status_code} - {response.text}"
        )

if __name__ == "__main__":
    print("Running MCP server on http://localhost:8002/mcp/")
    mcp.run(transport="streamable-http")