--
-- Script that lists all shows contained in hbtn_0d_tvshows
-- that have at least one genre linked.
-- Each record should display: tv_shows.title - tv_show_genres.genre_id
-- Results must be sorted in ascending order by tv_shows.title and
-- tv_show_genres.genre_id
-- You can use only one SELECT statement
-- The database name will be passed as an argument of the mysql command
--
SELECT `ts`.`title`, `tg`.`id` AS `genre_id`
FROM (`tv_shows` AS `ts`, `tv_genres` AS `tg`)
INNER JOIN `tv_show_genres` AS `tsg`
ON (`ts`.`id` = `tsg`.`show_id`) WHERE (`tg`.`id` = `tsg`.`genre_id`)
ORDER BY `ts`.`title`, `tsg`.`genre_id` ASC;
