"""LLM caching and logging system for Gemini API calls."""

import json
import hashlib
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

class LLMCache:
    """Caching and logging system for LLM API calls."""
    
    def __init__(self, cache_dir: str = "llm_cache", log_dir: str = "llm_logs"):
        self.cache_dir = Path(cache_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.cache_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Log file for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_file = self.log_dir / f"session_{timestamp}.log"
        
    def _create_cache_key(self, tweet_text: str, image_paths: Optional[List[str]], 
                         system_instruction: str) -> str:
        """Create a unique cache key for the input."""
        # Create hash from all inputs that affect the response
        content = {
            'tweet_text': tweet_text,
            'image_paths': sorted(image_paths) if image_paths else [],  # Handle None case
            'system_instruction': system_instruction
        }
        
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cached response."""
        return self.cache_dir / f"{cache_key}.json"
    
    def log_request(self, tweet_text: str, image_paths: List[str], 
                   system_instruction: str, cache_key: str, cache_hit: bool, tweet_id: str = None):
        """Log the LLM request details."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'tweet_id': tweet_id,
            'cache_key': cache_key,
            'cache_hit': cache_hit,
            'request': {
                'tweet_text': tweet_text,
                'image_paths': image_paths,
                'image_count': len(image_paths),
                'system_instruction': system_instruction,
                'system_instruction_length': len(system_instruction)
            }
        }
        
        # Write to session log
        with open(self.session_log_file, 'a', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write(f"🤖 GEMINI API REQUEST - {log_entry['timestamp']}\n")
            f.write("="*100 + "\n")
            f.write(f"🆔 Tweet ID: {tweet_id or 'N/A'}\n")
            f.write(f"🔑 Cache Key: {cache_key}\n")
            f.write(f"📋 Cache Hit: {cache_hit}\n")
            f.write(f"📝 Tweet Text:\n{tweet_text}\n\n")
            f.write(f"🖼️  Image Paths ({len(image_paths) if image_paths else 0}):\n")
            if image_paths:
                for i, path in enumerate(image_paths):
                    f.write(f"  {i+1}. {path}\n")
            else:
                f.write("  (no images)\n")
            f.write(f"\n🔧 System Instruction ({len(system_instruction)} chars):\n")
            f.write(f"Hash: {self._create_cache_key(tweet_text, image_paths or [], system_instruction)[:16]}...\n")
            f.write(f"{system_instruction}\n\n")
    
    def log_response(self, cache_key: str, response_text: str, 
                    parsed_result: Any, processing_time: float, error: str = None, tweet_id: str = None):
        """Log the LLM response details."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'tweet_id': tweet_id,
            'cache_key': cache_key,
            'processing_time': processing_time,
            'response_length': len(response_text),
            'error': error
        }
        
        # Write to session log
        with open(self.session_log_file, 'a', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write(f"📤 GEMINI API RESPONSE - {log_entry['timestamp']}\n")
            f.write("="*100 + "\n")
            f.write(f"🆔 Tweet ID: {tweet_id or 'N/A'}\n")
            f.write(f"🔑 Cache Key: {cache_key}\n")
            f.write(f"⏱️  Processing Time: {processing_time:.2f} seconds\n")
            f.write(f"📏 Response Length: {len(response_text)} characters\n")
            
            if error:
                f.write(f"❌ Error: {error}\n")
            else:
                f.write("✅ Success\n")
            
            f.write(f"\n📤 RAW RESPONSE:\n{response_text}\n\n")
            
            if parsed_result:
                f.write(f"📊 PARSED RESULT:\n{json.dumps(parsed_result, indent=2)}\n")
            else:
                f.write("❌ JSON PARSING FAILED\n")
            
            f.write("\n" + "="*100 + "\n\n")
    
    def get_cached_response(self, tweet_text: str, image_paths: List[str], 
                          system_instruction: str, tweet_id: str = None) -> Optional[Dict]:
        """Get cached response if available."""
        cache_key = self._create_cache_key(tweet_text, image_paths, system_instruction)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # Verify system instruction matches exactly (extra safety check)
                cached_system_instruction = cached_data.get('system_instruction')
                if cached_system_instruction != system_instruction:
                    print(f"⚠️  System instruction mismatch - cache invalidated")
                    print(f"   Expected length: {len(system_instruction)}")
                    print(f"   Cached length: {len(cached_system_instruction) if cached_system_instruction else 0}")
                    return None
                
                # Log cache hit
                self.log_request(tweet_text, image_paths, system_instruction, cache_key, cache_hit=True, tweet_id=tweet_id)
                
                print(f"🎯 CACHE HIT! Using cached response for key: {cache_key[:16]}...")
                
                return {
                    'cache_key': cache_key,
                    'cached': True,
                    **cached_data
                }
            except Exception as e:
                print(f"⚠️  Cache read error: {e}")
        
        return None
    
    def save_response(self, tweet_text: str, image_paths: List[str], 
                     system_instruction: str, response_text: str, 
                     parsed_result: Any, processing_time: float, tweet_id: str = None, thoughts: List[str] = None) -> str:
        """Save response to cache and return cache key."""
        cache_key = self._create_cache_key(tweet_text, image_paths, system_instruction)
        cache_path = self._get_cache_path(cache_key)
        
        # Prepare cache data
        cache_data = {
            'system_instruction': system_instruction,  # Store system instruction for verification
            'thoughts': thoughts,
            'result': parsed_result,
            'response_text': response_text,
            'cached_at': datetime.now().isoformat(),
            'input_text': tweet_text,
            'image_paths': image_paths,
            'image_count': len(image_paths) if image_paths else 0,
            'tweet_id': tweet_id,
            'processing_time': processing_time,
        }
        
        # Save to cache file
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        # Log the request and response
        self.log_request(tweet_text, image_paths, system_instruction, cache_key, cache_hit=False, tweet_id=tweet_id)
        self.log_response(cache_key, response_text, parsed_result, processing_time, tweet_id=tweet_id)
        
        print(f"💾 Response cached with key: {cache_key[:16]}...")
        
        return cache_key
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache."""
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'total_cached_responses': len(cache_files),
            'cache_size_mb': total_size / (1024 * 1024),
            'cache_directory': str(self.cache_dir),
            'log_directory': str(self.log_dir),
            'session_log_file': str(self.session_log_file)
        }
    
    def clear_cache(self):
        """Clear all cached responses."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        print("🗑️  Cache cleared")