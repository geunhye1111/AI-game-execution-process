import csv
import requests
from bs4 import BeautifulSoup
import re
from googlesearch import search

def get_keywords_from_search_results(game_name, user_keyword):
    # 검색어 생성
    query = f"{game_name} {user_keyword}"

    # 구글 검색에서 게임에 대한 결과 페이지 URL 가져오기
    search_results = search(query, num=10, stop=10, pause=2)  # 예를 들어, 상위 5개의 검색 결과만 사용
    keywords = []

    # 각 검색 결과 페이지에서 키워드 추출
    for url in search_results:
        try:
            # 웹 페이지에서 텍스트 가져오기
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()

            # 텍스트에서 단어 추출
            words = re.findall(r'\w+', text.lower())

            # 각 단어의 출현 빈도수 계산
            count = words.count(user_keyword.lower())
            keywords.append((game_name, count))
        except Exception as e:
            #print(f"Error occurred while processing URL: {url}")
            #print(e)
            continue

    return keywords

def recommend_game(games, user_keyword):
    game_counts = []
    for game in games:
        game_name = game.strip()
        keywords = get_keywords_from_search_results(game_name, user_keyword)
        total_count = sum(count for _, count in keywords)
        game_counts.append((game_name, total_count))

    # 빈도수가 가장 높은 게임 추천
    recommended_game = max(game_counts, key=lambda x: x[1])

    # 결과 출력
    print("게임별 키워드 언급 빈도수:")
    for game, count in game_counts:
        print(f"{game}: {count}")
    print(f"\n'{user_keyword}' 키워드를 가장 많이 언급한 게임: {recommended_game[0]} ({recommended_game[1]} mentions)")

# 게임 리스트
games = ["커넥트4", "오목", "simple grid", "tic-tac-toe", "Blackjack"]

# 사용자 입력 받기
user_keyword = input("키워드를 입력하세요: ")

# 게임 추천
recommend_game(games, user_keyword)