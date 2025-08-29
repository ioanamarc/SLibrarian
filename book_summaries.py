# db/book_summaries.py

book_summaries_dict = {
    "1984": (
        "George Orwell’s novel depicts a dystopian society under total state control. "
        "Citizens are constantly monitored by Big Brother, and independent thought is considered a crime. "
        "Winston Smith, the main character, tries to resist this oppressive regime. "
        "The story explores themes of freedom, truth, and ideological manipulation."
    ),
    "The Hobbit": (
        "Bilbo Baggins, a comfort-loving hobbit, is unexpectedly swept into an adventure "
        "to help dwarves reclaim their treasure guarded by the dragon Smaug. "
        "On his journey, Bilbo discovers courage and resourcefulness he never knew he had. "
        "The story is filled with fantasy creatures, unlikely friendships, and thrilling moments."
    ),
    "To Kill a Mockingbird": (
        "Harper Lee’s classic set in the racially segregated American South. "
        "The story follows young Scout Finch as her father, a lawyer, defends a black man falsely accused of assault. "
        "Themes include racial injustice, moral growth, and empathy."
    ),
    "The Catcher in the Rye": (
        "Holden Caulfield narrates a few days in his life after being expelled from prep school. "
        "The novel explores teenage angst, alienation, and the struggle to find authenticity in a superficial world."
    ),
    "Brave New World": (
        "Aldous Huxley imagines a future society obsessed with technology, pleasure, and conformity. "
        "Humanity is conditioned from birth to accept their roles without question. "
        "Themes include freedom, individuality, and the cost of utopia."
    ),
    "Fahrenheit 451": (
        "In a future where books are banned, fireman Guy Montag burns them until he begins to question the system. "
        "This novel explores censorship, the power of knowledge, and rebellion against conformity."
    ),
    "The Great Gatsby": (
        "Jay Gatsby throws extravagant parties in pursuit of Daisy Buchanan, a symbol of lost love and the American Dream. "
        "Set in the Jazz Age, the novel explores wealth, illusion, and the hollowness of success."
    ),
    "Harry Potter and the Sorcerer's Stone": (
        "Harry Potter, an orphan raised by his unkind relatives, discovers he is a wizard and attends Hogwarts School. "
        "He makes friends, uncovers secrets, and confronts evil. "
        "Themes include friendship, magic, and courage."
    ),
    "Animal Farm": (
        "Orwell’s allegory of the Russian Revolution, where farm animals overthrow their human owner to form a society of equals. "
        "However, power corrupts, and a new tyranny emerges. "
        "Themes include propaganda, power, and betrayal."
    ),
    "Lord of the Flies": (
        "A group of boys stranded on an uninhabited island descend into savagery as their attempts at order fail. "
        "The novel explores the fragility of civilization and the darkness within human nature."
    )
}


def get_summary_by_title(title: str) -> str:
    """Returns the detailed summary for an exact book title."""
    return book_summaries_dict.get(title, "Sorry, this book is not in the database.")
