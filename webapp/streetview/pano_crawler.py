"""Crawl Google Street View panoramas around a given lat/lng using the streetlevel library."""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import aiohttp
from streetlevel import streetview


@dataclass
class PanoInfo:
    id: str
    lat: float
    lng: float
    heading: float
    date: str
    image_path: str


async def _download_pano(pano, output_dir: Path, session: aiohttp.ClientSession) -> PanoInfo | None:
    """Download a single panorama image and return its info."""
    try:
        image_path = output_dir / f"{pano.id}.jpg"
        await streetview.download_panorama_async(pano, str(image_path), session, zoom=3)
        return PanoInfo(
            id=pano.id,
            lat=pano.lat,
            lng=pano.lon,
            heading=pano.heading if pano.heading is not None else 0.0,
            date=str(pano.date) if pano.date else "",
            image_path=str(image_path),
        )
    except Exception as e:
        print(f"Warning: failed to download pano {pano.id}: {e}", file=sys.stderr)
        return None


async def crawl_panos(
    lat: float,
    lng: float,
    output_dir: Path,
    max_panos: int = 50,
) -> list[PanoInfo]:
    """BFS crawl of Street View panoramas near the given coordinates.

    Returns a list of PanoInfo for each successfully downloaded panorama.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        # Find the closest panorama to our starting point
        start_pano = await streetview.find_panorama_async(lat, lng, session)
        if start_pano is None:
            raise RuntimeError(f"No Street View coverage found near ({lat}, {lng})")

        visited: set[str] = set()
        queue: list = [start_pano]
        panos: list[PanoInfo] = []

        while queue and len(visited) < max_panos:
            current = queue.pop(0)
            if current.id in visited:
                continue
            visited.add(current.id)

            info = await _download_pano(current, output_dir, session)
            if info:
                panos.append(info)
                print(f"Downloaded {len(panos)}/{max_panos}: {current.id}")

            # Enqueue neighbors via links
            if current.links:
                for link in current.links:
                    neighbor_pano = link.pano
                    if neighbor_pano and neighbor_pano.id not in visited:
                        try:
                            # Fetch full metadata for the neighbor
                            full_neighbor = await streetview.find_panorama_by_id_async(
                                neighbor_pano.id, session
                            )
                            if full_neighbor:
                                queue.append(full_neighbor)
                        except Exception:
                            pass

    if len(panos) < 5:
        raise RuntimeError(
            f"Only found {len(panos)} panoramas (minimum 5 required). "
            "The area may have limited Street View coverage."
        )

    # Write metadata
    metadata = [asdict(p) for p in panos]
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Crawled {len(panos)} panoramas, metadata saved to {metadata_path}")

    return panos


def main():
    parser = argparse.ArgumentParser(description="Crawl Street View panoramas")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lng", type=float, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_panos", type=int, default=50)
    args = parser.parse_args()

    asyncio.run(crawl_panos(args.lat, args.lng, Path(args.output), args.max_panos))


if __name__ == "__main__":
    main()
