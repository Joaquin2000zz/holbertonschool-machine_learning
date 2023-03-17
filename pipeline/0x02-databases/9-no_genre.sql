--
-- script that lists all shows contained in hbtn_0d_tvshows
-- without a genre linked.
-- 
-- - Each record should display: tv_shows.title - tv_show_genres.genre_id
-- - Results must be sorted in ascending order by tv_shows.title
-- and tv_show_genres.genre_id
-- - You can use only one SELECT statement
-- - The database name will be passed as an argument of the mysql command
--
SELECT t1.title, t2.genre_id AS genre_id
FROM (tv_shows AS t1)
LEFT JOIN tv_show_genres AS t2
ON (t1.id = t2.show_id) WHERE (t2.show_id IS NULL)
ORDER BY t1.title ASC;
