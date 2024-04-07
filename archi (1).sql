-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Apr 07, 2024 at 04:03 PM
-- Server version: 10.4.28-MariaDB
-- PHP Version: 8.2.4

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `archi`
--

-- --------------------------------------------------------

--
-- Table structure for table `translations`
--

CREATE TABLE `translations` (
  `ancient_word` varchar(255) NOT NULL,
  `english_translation` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Dumping data for table `translations`
--

INSERT INTO `translations` (`ancient_word`, `english_translation`) VALUES
('ali-pavatahi pugiyana line sagasa', 'The cave of the members of the Corporation in Alipavata is given to the Sangha'),
('anikata-sona-pitaha bariya upasika-tisaya lene agata-anagata-catu-disa-sagasa', 'The cave of the female lay-devotee Tissa, wife of teh father of Sona, the Cavalryman is given to the Sangha of the four quaters, present and absent'),
('asi-pusagutasa', 'Of the venerable Phussagutta'),
('bamana-suga-puta-pusadevasa cs datasa ca lene sagasa', 'The cave of Phussadeva, the son of the Brahmana Sunga is given to the Sangha'),
('barata-mahatisaha lene catu-disa-sagasa niyate', 'The cave of lord Mahatissa is dedicated to the Sangha of the four quarters'),
('barata-mahatisaha lene sagasa niyate', 'The cave of lord Mahatissa is dedicated to the Sangha'),
('barata-tisa-gutaha lene sagasa', 'The cave of lord Tissagutta is given to the Sangha'),
('bata-cudatisaha lene', 'The cave of lord Culatissa'),
('bata-devagutaha batikaha gapati-samanaha lene', 'The cave of the householder Samana, brother of lord Devagutta'),
('bata-mahadata manapadasane', 'The cave Manapadassana of Lord Mahadatta'),
('bata-nagaha lene', 'The cave of lord Naga'),
('bata-nagaha lene sagasa', 'The cave of Lord Naga is given to the Sangha'),
('bata-nagaha lene sagasa mataligamika-puta gamika-tisaha lene', 'The cave of lord Naga is given to the Sangha. The cave of the village-councillor Tissa, son of the village-councillor of Mataligama'),
('bata-palaha lene', 'The cave of lord Pala'),
('bata-punasa lene sagaye dine', 'The cave of lord Punna is given to the Sangha'),
('bata-pusabutiya lene sagasa', 'The cave of lord Phussabhuti is given to the Sangha'),
('bata-rakiya lene', 'The cave of lord Raki'),
('bata-sataha lene parumaka-abaha lene sagasa agata-anagata-catudisa-sagasa', 'The cave of the Lord Sata and the cave of the chief Abhaya are given to the Sangha of the four quaters, present and absent '),
('bata-sudinasa lene sagasa', 'The cave of lord Sudinna is given to the Sangha'),
('bata-sumanabutiya lene', 'The cave of lord Sumanabhuti'),
('bata-sumanagutaha lene', 'The cave of Lord Sumanagutta'),
('bata-surigutaha lene', 'The cave of lord Suriyagutta'),
('bata-tisagotasa', 'Of lord Tissagotta'),
('bata-tisasa lene agata-anagata-catu-disa-sagasa dine', 'The cave of lord Tissa is given to the Sangha of the four quarters, present and absent'),
('bata-vasuha lene', 'The cave of lord Vadu'),
('bata-visadevaha lene', 'The cave of lord Visadeva'),
('batu-cudi-tisaha lene', 'The cave of the lord Cula-Tissa'),
('bhata sonasa lene maha-sudasane catu-disa-saghasa patithite', 'The cave named Maha-Sudassana of lord Sona has been established for the benefit of the Sangha of the four quarters'),
('budarakita-terasa supaditi', 'The cave named suppatittha, of the elder Buddharakkhita'),
('budataha lene', 'The cave of Buddhadatta'),
('cuda-sonaha mata upasika rakiya lene', 'The cave of the female lay-devotee Raki, the mother of Culla-Sona'),
('damarakita-terasa', 'Of the elder Dhammarakkhita'),
('data-teraha lene dana sagasa', 'The cave of elder Datta is a gift to the Sangha'),
('Devanapiya-maharajaha gamini-tisaha puta mahatisa-ayaha lene sagike', 'The cave of prince Mahatissa, son of the great king Gamani Tissa, the friend of Gods is the property of the Sangha'),
('devarakitasa ca damaterasa ca lene', 'The cave of Devarakkhita and of the elder Dhamma'),
('eka-utirika-banaka pusadeva-teraha manorame', 'The cave named Manorama of the elder Phussadeva, the Reciter of Ekottarika'),
('gamani-dhamarajhasa putasa aya-asalisa lene', 'The cave of the prince Asali, son of Gamani Dhammaraja'),
('gamika-duhatara-puta gamika-abayaha lene sagasa niyate', 'The cave of the village-councillor Abhaya, son of the village-councillor Duharata is dedicated to the Sangha'),
('gamika-tina lene', 'The cave of village councillor Tina'),
('gamika-tisaha lene', 'The cave of the village-councillor Tissa'),
('gapati-retasa lene ya', 'This is the cave of the householder Reta'),
('gapati-topasa-sumana-kulasa lene sagasa dine agata-anagata-catu-disa-sagasa pasu visaraye', 'The cave of the family of the householder Sumana the tinsmith is given to the Sangha as comfortable abode of the Sangha of the four quarters, present and absent'),
('gapati-velu-putana tini-batikana sagasa bata-dataha lene', 'The cave of the three brothers, the sons of the householder Velu is given to the Sangha. The cave of Lord Datta'),
('guta-putaha parumaka-puraha lene agat-agata catu-disa-sagasa', 'The cave of the chief Pura, son of Gutta is given to the Sangha to all that are come from the four quarters'),
('kabala-tisaha lene sagasa', 'The cave of Kambala-Tissa is given to the Sangha'),
('maha-samuda-puta-gutasa lene sagasa parumaka-bamadata-putaha mahagutaha lene', 'The cave of Gutta, son of Maha-Samudda is given to the Sangha. The cave of Mahagutta, son of the chief Brahmadatta'),
('mahayaha putaha arake', 'The protection of the son of the heir-apparent'),
('Manikara-Mulagutaha padagadini', 'The steps of the lapidry Mulagutta'),
('manorame lene bata-gutena karite sagasa', 'The cave named Manorama caused to be founded by lord Gutta is given to the Sangha'),
('matula-baginiyana lene agata-anagata-catu disa-sagasa niyata se', 'The cave of the uncle and nephew is dedicated to the Sangha of the four quarters, present and absent'),
('pakara-adeka samudaha lene sagasa', 'The cave of Samudda, the Superintendent of Roads is given to the Sangha'),
('paladata-tisaha lene', 'The cave of Phaladatta Tissa'),
('parumaka-abijhiya lene cuda-tisaha lene sagaye niyate', 'The cave of the chief Abhijhi and the cave of Culla-Tissa are dedicated tot he Sangha'),
('parumaka-bagu-tisaha lene sagasa dine', 'The cave of the chief Bhaga-Tissa is given to the Sangha'),
('parumaka-bama-puta parumaka-tisaha lene sagasa', 'The cave of the chief Tissa, the son of teh chief Bhama is given to the Sangha'),
('parumaka-canisata-sumana-putaha parumaka-patakana-satasa sagasa', 'The cave of the chief Patakana Sata, son of the chief Canisata Sumana is given to the Sangha'),
('parumaka-danamitasa lene', 'The cave of the chief Dhanamitta'),
('parumaka-devanakatasa puta acariya-kanadatasa', 'Of the teacher Kanhadatta, son of the chief Devanakkhatta'),
('parumaka-dinaha lene sagasa', 'The cave of the chief Dinna is given to the Sangha'),
('parumaka-dumana-puta parumaka-hagaraha lene agata-anagata-caru-disa-sagasa dine', 'The cave of the chief Hagara, son of the chief Sumana is given to the Sangha of the four quarters, present and absent'),
('parumaka-kutaragaya-veluha lene', 'The cave of the chief Velu, the householder of the Vase'),
('parumaka-maharetasa puta parumaka-nagasa lene agata-anagata-catu-disa-sagasa dine', 'The cave of the chief Naga, son of the chief Mahareta, is given to the Sangha of the four quarters, present and absent'),
('parumaka-naga-puta parumaka-pigala-gutasa lene agata-anagata-catu-disa-saghasa dine', 'The cave of the chief Pinagalagutta, soon of the chief Naga is given to the Sangha of the four quaters, present and absent'),
('Parumaka-naga-puta-asaliya lene agata-anagata-catudisika-sagaye', 'The cave of Asali, son of the chief Naga is given to the Sangha of the four quarters, present and absent'),
('parumaka-naga-puta-tisaha lene sagasa', 'The cave of Tissa, son of the chief Naga, is granted to the Sangha'),
('parumaka-namali-sumana-puta parumaka-namaliya lene sagasa', 'The cave of the chief Namali, son of the chief Namali Sumana is given to the Sangha'),
('parumaka-palikadasa bariya parumaka-surakita-jhita upasika-citaya lene sagasa catu-disa', 'The cave of the female lay-devotee Citta, wife of the chief Palikada and daughter of the chief Surakkhita is given to the Sangha of the four quaters'),
('parumaka-raka-puta parumaka-paraka asaha lene', 'The cave of the chief Paraka Asa, son of the chief Rakkha'),
('parumaka-raki-puta-mahatisaha lene sagasa', 'The cave of Mahatissa, son of the chief Raki is given to the Sangha'),
('parumaka-samudaha lene agata-anagata sagasa', 'The cave of the chief Samudda given to the Sangha, present and absent'),
('parumaka-sena-puta badakarika-parumaka-senaha lene sagasa', 'The cave of the chief Sena, the treasurer, son of the chief Sena is given to the Sangha'),
('parumaka-senaha lene sagasa', 'The cave of the chief Sena is given to the Sangha'),
('parumaka-suridaha lene', 'The cave of the chief Surinda'),
('parumaka-tisa-putaha lene sagasa dutakaha', 'The cave of the son of the chief Bhama is given to the Sangha'),
('parumaka-velaha lene sagasa', 'The cave of the chief Vela is given to the Sangha'),
('pusaha bata-nagasa mahacitasa', 'Of Phussa, of lord Naga, of Mahacitta'),
('sagasa sisapane', 'Given to the Sangha is the cave named Sithhaphana'),
('sagasa upasaka-cudaha asiya sivaha sadi sadaya', 'The cave given to the Sangha of the lay-devotee Cuja in co-partnersip with venerable Siva'),
('sumanagutaha lene', 'The cave of Sumanagutta'),
('taladara-nagaya-puta-devaha lene agata-angata-catu-disa-sagasa', 'The cave of Deva, son of Nagaya the GoldSmith is given to the Sangha of the four quaters, present or absent'),
('tatavaya-pugaha lene', 'The cave of the weavers corperation'),
('tisa-samaniya lene sagasa', 'The cave of the nun Tissa is given to the sangha'),
('tisa-teraha lene sagasa', 'The cave of the elder Tissa is given to the Sangha'),
('tubada-vasaka-pugiyana lene', 'The cave of the members of the Corporation of the Tubada family'),
('uparajha-naga-pute rajha-abaya nama tasa pute gamani-tise nama tena karite sudasane sagasa', 'The son of Uparaja Naga was king Abaya by name. His son is Gamani Tissa by name. The cave named Sudassana founded by him is given to the Sangha'),
('upasaka-cudaha lene sagasa', 'The cave of the lay-devotee Cuja, is given to the Sangha'),
('upasaka-devaha lene', 'The cave of the female lay-devotee Deva'),
('upasaka-kacaliya lene', 'The cave of the lay-devotee Kacali'),
('upasaka-nagaha lene sagasa dine', 'The cave of the lay-devotee Naga, is given to the Sangha'),
('upasaka-rakiya lene', 'The cave of lay-devotee Raki'),
('upasaka-salaha lene', 'The cave of the lay-devotee Sala'),
('upasaka-samudaha lene sagasa', 'The cave of the lay-devotee Samudda is given to the Sangha'),
('upasika somaliya lene', 'The cave of the female lay-devotee Somali'),
('upasika-purusadataya lene', 'The cave of the family lay-devotee Purisadatta'),
('upasika-tisaya lene', 'The cave of the female lay-devotee Tissa');

-- --------------------------------------------------------

--
-- Table structure for table `words`
--

CREATE TABLE `words` (
  `num` int(10) NOT NULL,
  `brahmi` varchar(15) NOT NULL,
  `english` varchar(15) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `words`
--

INSERT INTO `words` (`num`, `brahmi`, `english`) VALUES
(4, 'prumk', 'parumaka'),
(6, 'shgsh', 'shagasha'),
(7, 'leNhe', 'lene'),
(8, 'angth', 'anagatha'),
(9, 'dhish', 'disha'),
(10, 'agth', 'agatha'),
(11, 'puth', 'putha');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `translations`
--
ALTER TABLE `translations`
  ADD PRIMARY KEY (`ancient_word`);

--
-- Indexes for table `words`
--
ALTER TABLE `words`
  ADD PRIMARY KEY (`num`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `words`
--
ALTER TABLE `words`
  MODIFY `num` int(10) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=12;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
