--
-- script that ranks country origins of bands, ordered by the number of (non-unique) fans
-- 
-- - Import this table dump: metal_bands.sql.zip
-- - Column names must be: origin and nb_fans
-- - Your script can be executed on any database
-- Context: Calculate/compute something is always power intensive… better to distribute the load!
--
SELECT `origin`, SUM(`fans`) AS `nb_fans`
FROM `metal_bands` AS `mb`
WHERE `origin` IS NOT NULL AND `origin` = `mb`.`origin`
GROUP BY `origin`
ORDER BY `nb_fans` DESC;
