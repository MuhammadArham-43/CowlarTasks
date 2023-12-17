from model import get_recommendation

if __name__ == "__main__":
    print("-- MOVIE RECOMMENDATION SYSTEM --")
    print("Enter movie name or q to quit.\n")
    
    while True:
        movie = input(">> ")
        if movie.lower() == "q":
            exit(0)
        recommendation = get_recommendation(movie)
        print(f"WATCH NEXT:  {', '.join(recommendation)}" if isinstance(recommendation, list) else recommendation)