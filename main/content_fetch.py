"""
Content Fetcher using TMDB API
Retrieves real TV shows and movies based on BN recommendations
"""

import os
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TMDBContentFetcher:
    """
    Fetches TV content from TMDB API based on program type and genre recommendations
    """
    
    BASE_URL = "https://api.themoviedb.org/3"
    
    # Mapping from BN genres to TMDB genre IDs
    GENRE_MAPPING = {
        # Movies
        "comedy": {"movie": 35, "tv": 35},
        "drama": {"movie": 18, "tv": 18},
        "horror": {"movie": 27, "tv": 9648},  # Mystery for TV
        "romance": {"movie": 10749, "tv": 10749},
        "action": {"movie": 28, "tv": 10759},  # Action & Adventure for TV
        "thriller": {"movie": 53, "tv": 9648},
        "sci-fi": {"movie": 878, "tv": 10765},  # Sci-Fi & Fantasy for TV
        "fantasy": {"movie": 14, "tv": 10765},
        
        # TV Specific
        "documentary": {"movie": 99, "tv": 99},
        "news": {"tv": 10763},
        "entertainment": {"tv": 10764},  # Reality
        "talk": {"tv": 10767},
        "animation": {"movie": 16, "tv": 16}
    }
    
    # Mapping from BN program types to TMDB media types
    TYPE_MAPPING = {
        "movie": "movie",
        "series": "tv",
        "documentary": "tv",  # TMDB treats documentaries as TV
        "news": "tv",
        "entertainment": "tv"
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TMDB client
        
        Args:
            api_key: TMDB API key (if None, reads from environment variable TMDB_API_KEY)
        """
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        if not self.api_key:
            raise ValueError("TMDB API key not provided. Set TMDB_API_KEY environment variable.")
        
        self.session = requests.Session()
        self.session.params = {"api_key": self.api_key}
    
    def _get(self, endpoint: str, **params) -> Dict:
        """Make GET request to TMDB API"""
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_content_by_recommendation(
        self,
        program_type: str,
        program_genre: str,
        limit: int = 10,
        language: str = "es-ES"
    ) -> List[Dict]:
        """
        Fetch content based on BN recommendation
        
        Args:
            program_type: Type from BN (movie, series, documentary, etc.)
            program_genre: Genre from BN (comedy, drama, etc.)
            limit: Maximum number of results
            language: Language for content (default: Spanish)
            
        Returns:
            List of content items with title, overview, genre, etc.
        """
        
        # Map to TMDB types
        media_type = self.TYPE_MAPPING.get(program_type, "tv")
        
        # Get genre ID
        genre_id = None
        if program_genre in self.GENRE_MAPPING:
            genre_id = self.GENRE_MAPPING[program_genre].get(media_type)
        
        logger.info(f"Fetching {media_type} content for genre: {program_genre} (ID: {genre_id})")
        
        # Discover content
        results = self._discover_content(
            media_type=media_type,
            genre_id=genre_id,
            limit=limit,
            language=language
        )
        
        # Format results
        formatted = []
        for item in results:
            formatted.append(self._format_item(item, media_type))
        
        return formatted
    
    def _discover_content(
        self,
        media_type: str,
        genre_id: Optional[int] = None,
        limit: int = 10,
        language: str = "es-ES"
    ) -> List[Dict]:
        """
        Discover content using TMDB discover endpoint
        """
        endpoint = f"discover/{media_type}"
        
        params = {
            "language": language,
            "sort_by": "popularity.desc",
            "page": 1,
            "vote_count.gte": 10  # Filter out very obscure content
        }
        
        if genre_id:
            params["with_genres"] = genre_id
        
        # For TV shows, prefer currently airing or recent
        if media_type == "tv":
            # Get date 2 years ago
            two_years_ago = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
            params["first_air_date.gte"] = two_years_ago
        
        try:
            data = self._get(endpoint, **params)
            results = data.get("results", [])[:limit]
            return results
        except Exception as e:
            logger.error(f"Error fetching content: {e}")
            return []
    
    def _format_item(self, item: Dict, media_type: str) -> Dict:
        """
        Format TMDB item to our internal structure
        """
        if media_type == "movie":
            title = item.get("title", "")
            date = item.get("release_date", "")
        else:
            title = item.get("name", "")
            date = item.get("first_air_date", "")
        
        return {
            "id": item.get("id"),
            "title": title,
            "overview": item.get("overview", ""),
            "type": media_type,
            "genre_ids": item.get("genre_ids", []),
            "popularity": item.get("popularity", 0),
            "vote_average": item.get("vote_average", 0),
            "vote_count": item.get("vote_count", 0),
            "release_date": date,
            "poster_path": item.get("poster_path"),
            "backdrop_path": item.get("backdrop_path"),
            "original_language": item.get("original_language", "")
        }
    
    def search_content(
        self,
        query: str,
        media_type: str = "multi",
        language: str = "es-ES"
    ) -> List[Dict]:
        """
        Search for specific content by name
        
        Args:
            query: Search query
            media_type: Type to search (movie, tv, multi)
            language: Language for results
            
        Returns:
            List of matching content
        """
        endpoint = f"search/{media_type}"
        
        try:
            data = self._get(endpoint, query=query, language=language)
            results = data.get("results", [])
            
            formatted = []
            for item in results:
                item_type = item.get("media_type", media_type)
                if item_type in ("movie", "tv"):
                    formatted.append(self._format_item(item, item_type))
            
            return formatted
        except Exception as e:
            logger.error(f"Error searching content: {e}")
            return []
    
    def get_trending(
        self,
        media_type: str = "all",
        time_window: str = "day",
        language: str = "es-ES"
    ) -> List[Dict]:
        """
        Get trending content
        
        Args:
            media_type: Type (all, movie, tv)
            time_window: Time window (day, week)
            language: Language for results
            
        Returns:
            List of trending content
        """
        endpoint = f"trending/{media_type}/{time_window}"
        
        try:
            data = self._get(endpoint, language=language)
            results = data.get("results", [])
            
            formatted = []
            for item in results:
                item_type = item.get("media_type", media_type)
                if item_type in ("movie", "tv"):
                    formatted.append(self._format_item(item, item_type))
            
            return formatted
        except Exception as e:
            logger.error(f"Error getting trending content: {e}")
            return []
    
    def get_content_details(
        self,
        content_id: int,
        media_type: str,
        language: str = "es-ES"
    ) -> Dict:
        """
        Get detailed information about a specific content item
        """
        endpoint = f"{media_type}/{content_id}"
        
        try:
            data = self._get(endpoint, language=language)
            return data
        except Exception as e:
            logger.error(f"Error getting content details: {e}")
            return {}
    
    def get_genres_list(self, media_type: str = "movie", language: str = "es-ES") -> List[Dict]:
        """
        Get list of all genres from TMDB
        """
        endpoint = f"genre/{media_type}/list"
        
        try:
            data = self._get(endpoint, language=language)
            return data.get("genres", [])
        except Exception as e:
            logger.error(f"Error getting genres: {e}")
            return []


def select_best_match(
    candidates: List[Dict],
    preferences: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Select the best content from candidates based on preferences and quality metrics
    
    Args:
        candidates: List of content items
        preferences: Optional dict with user preferences (min_rating, etc.)
        
    Returns:
        Best matching content item
    """
    if not candidates:
        return None
    
    # Default preferences
    min_rating = preferences.get("min_rating", 6.0) if preferences else 6.0
    min_votes = preferences.get("min_votes", 50) if preferences else 50
    
    # Filter by quality
    filtered = [
        c for c in candidates
        if c.get("vote_average", 0) >= min_rating and c.get("vote_count", 0) >= min_votes
    ]
    
    # If too restrictive, relax filters
    if not filtered:
        filtered = candidates
    
    # Sort by popularity and rating
    filtered.sort(
        key=lambda x: (x.get("popularity", 0) * 0.6 + x.get("vote_average", 0) * 0.4),
        reverse=True
    )
    
    return filtered[0]


# Example usage
if __name__ == "__main__":
    # Example: Get comedy movies
    fetcher = TMDBContentFetcher()
    
    # Based on BN recommendation
    content = fetcher.get_content_by_recommendation(
        program_type="movie",
        program_genre="comedy",
        limit=5
    )
    
    for item in content:
        print(f"- {item['title']} ({item['vote_average']}/10)")
        print(f"  {item['overview'][:100]}...")
        print()