# Movie Recommendation System

Install dependencies from the requirements file in the root directory.

```
pip install -r requirements.txt
```

Run the recommendation model using

```
python main.py
```

## Preprocessing

The recommendations are scaled to 0-1 for better accuracy enabling model to work for different recommendation scales.

## Model Selection

We use simple Cosine Similarity Matching for finding the most similar rating vectors.

Given a movie name, we fetch its ranking values row and compare it to every movie ratings in the dataset. The top-k with the highest Cosine Similarity score are returned.

## Results

> > Pirates of the Caribbean: Dead Man's Chest (2006)
> > WATCH NEXT: Pirates of the Caribbean: At World's End (2007)

> > Rocky (1976)
> > WATCH NEXT: RoboCop (1987)

> > Snow White and the Seven Dwarfs (1937)
> > WATCH NEXT: Pinocchio (1940)

> > Star Wars: Episode I - The Phantom Menace (1999)
> > WATCH NEXT: Total Recall (1990)

> > Terminator 2: Judgment Day (1991)
> > WATCH NEXT: Jurassic Park (1993)

> > Toy Story 3 (2010)
> > WATCH NEXT: Ratatouille (2007)

> > Batman Begins (2005)
> > WATCH NEXT: Dark Knight, The (2008)
