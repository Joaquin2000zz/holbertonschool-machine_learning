--
-- script that lists all genres from hbtn_0d_tvshows_rate by their rating.
-- Each record should display: tv_genres.name - rating sum
-- Results must be sorted in descending order by their rating
-- You can use only one SELECT statement
-- The database name will be passed as an argument of the mysql command
--
SELECT g.name, SUM(r.rate) AS rating
FROM tv_show_ratings as r
INNER JOIN tv_show_genres AS sg ON (r.show_id = sg.show_id)
INNER JOIN tv_genres AS g ON (g.id = sg.genre_id)
WHERE (r.rate IS NOT NULL)
GROUP BY g.name ORDER BY rating DESC;
