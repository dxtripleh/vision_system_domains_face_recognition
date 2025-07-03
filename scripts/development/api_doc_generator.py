#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API Documentation Generator.

FastAPI 기반 얼굴인식 API의 문서를 자동 생성합니다.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
import requests
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class APIDocumentationGenerator:
    """API 문서 자동 생성기"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.output_dir = Path("docs/api")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_docs(self):
        """모든 문서 생성"""
        logger.info("API 문서 생성 시작...")
        
        try:
            # OpenAPI 스키마 가져오기
            schema = self._fetch_openapi_schema()
            
            if schema:
                # 다양한 형식으로 문서 생성
                self._generate_markdown_docs(schema)
                self._generate_postman_collection(schema)
                
                logger.info("API 문서 생성 완료")
            else:
                logger.error("OpenAPI 스키마를 가져올 수 없습니다")
                
        except Exception as e:
            logger.error(f"문서 생성 중 오류 발생: {str(e)}")
    
    def _fetch_openapi_schema(self) -> Dict[str, Any]:
        """OpenAPI 스키마 가져오기"""
        try:
            response = requests.get(f"{self.api_base_url}/openapi.json", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"API 서버에서 스키마를 가져올 수 없음: {str(e)}")
            return {}
    
    def _generate_markdown_docs(self, schema: Dict[str, Any]):
        """Markdown 문서 생성"""
        logger.info("Markdown 문서 생성 중...")
        
        content = f"""# Face Recognition API Documentation

API 버전: {schema.get('info', {}).get('version', 'Unknown')}
생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 개요

{schema.get('info', {}).get('description', '얼굴인식 API 서비스')}

## 기본 정보

- **Base URL**: `{self.api_base_url}`
- **API 버전**: {schema.get('info', {}).get('version', 'v1')}

## 엔드포인트

"""
        
        # 경로별 문서 생성
        paths = schema.get('paths', {})
        for path, methods in paths.items():
            content += f"\n### {path}\n\n"
            
            for method, details in methods.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'DELETE']:
                    content += f"#### {method.upper()} {path}\n\n"
                    content += f"**설명**: {details.get('summary', '설명 없음')}\n\n"
        
        output_file = self.output_dir / "api_documentation.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Markdown 문서 생성 완료: {output_file}")
    
    def _generate_postman_collection(self, schema: Dict[str, Any]):
        """Postman 컬렉션 생성"""
        logger.info("Postman 컬렉션 생성 중...")
        
        collection = {
            "info": {
                "name": "Face Recognition API",
                "description": schema.get('info', {}).get('description', ''),
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "item": []
        }
        
        output_file = self.output_dir / "Face_Recognition_API.postman_collection.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(collection, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Postman 컬렉션 생성 완료: {output_file}")


if __name__ == "__main__":
    generator = APIDocumentationGenerator()
    generator.generate_all_docs() 